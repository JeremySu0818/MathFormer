import json
import math
import struct
import os
from typing import List, Dict, Tuple, Optional, Any, Union
from array import array
from itertools import chain

import operator

# Pre-bind operators and math functions for speedup
_add = operator.add
_sub = operator.sub
_mul = operator.mul
_math_exp = math.exp
_math_cos = math.cos
_math_sin = math.sin
_math_sqrt = math.sqrt


def vec_add(a, b):
    return list(map(_add, a, b))

def vec_sub(a, b):
    return list(map(_sub, a, b))

def vec_mul_scalar(v, s):
    return [x * s for x in v]

def vec_elem_mul(a, b):
    return list(map(_mul, a, b))

def mat_vec_mul(W, x):
    # sum(map(mul, row, x)) pushes inner loop to C
    return [sum(map(_mul, row, x)) for row in W]

def softmax(x):
    max_val = max(x)
    exps = [_math_exp(val - max_val) for val in x]
    sum_exps = sum(exps)
    inv_sum = 1.0 / sum_exps
    return [e * inv_sum for e in exps]

def silu(x):
    return x / (1.0 + _math_exp(-x))

def rms_norm(x, w, eps):
    sum_sq = sum(v * v for v in x)
    scale = 1.0 / _math_sqrt(sum_sq / len(x) + eps)
    return [val * scale * weight for val, weight in zip(x, w)]


def load_safetensors(path: str) -> Dict[str, Any]:
    """Optimized safetensors loader: read entire file once, then slice via memoryview."""
    with open(path, "rb") as f:
        file_data = f.read()  # Read entire file into memory at once

    header_size = struct.unpack_from("<Q", file_data, 0)[0]
    header = json.loads(file_data[8:8 + header_size])
    data_start = 8 + header_size

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue

        offsets = info["data_offsets"]
        start = data_start + offsets[0]
        end = data_start + offsets[1]

        shape = info["shape"]
        dtype = info["dtype"]

        if dtype == "F32":
            num_elements = (end - start) // 4
            # struct.unpack from buffer directly, no intermediate copy
            raw_data = struct.unpack_from(f"<{num_elements}f", file_data, start)
        elif dtype == "BF16" or dtype == "F16":
            raise NotImplementedError(f"Dtype {dtype} not implemented in pure python reader yet")
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        # Optimized reshape: for 1D, just convert tuple->list; for 2D, use slicing
        if len(shape) == 1:
            tensors[name] = list(raw_data)
        elif len(shape) == 2:
            rows, cols = shape
            # Slice the flat tuple into rows - much faster than recursive reshape
            tensors[name] = [list(raw_data[r * cols:(r + 1) * cols]) for r in range(rows)]
        else:
            # Fallback for higher dimensions (unlikely for this model)
            def reshape(data_iter, dims):
                if len(dims) == 1:
                    return [next(data_iter) for _ in range(dims[0])]
                return [reshape(data_iter, dims[1:]) for _ in range(dims[0])]
            data_iter = iter(raw_data)
            tensors[name] = reshape(data_iter, shape)

    return tensors


class Linear:
    __slots__ = ('weight', 'bias')

    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        out = [sum(map(_mul, row, x)) for row in self.weight]
        if self.bias:
            out = list(map(_add, out, self.bias))
        return out


class RMSNorm:
    __slots__ = ('weight', 'eps')

    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        sum_sq = sum(v * v for v in x)
        scale = 1.0 / _math_sqrt(sum_sq / len(x) + self.eps)
        return [val * scale * w for val, w in zip(x, self.weight)]


class LlamaRotaryEmbedding:
    __slots__ = ('dim', 'cos_cached', 'sin_cached')

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        self.dim = dim
        inv_freq = [1.0 / (base ** (i / dim)) for i in range(0, dim, 2)]

        # Pre-compute and store as tuples (immutable, faster iteration)
        cos_cached = []
        sin_cached = []
        for pos in range(max_position_embeddings):
            cos_vals = []
            sin_vals = []
            for freq in inv_freq:
                val = pos * freq
                cos_vals.append(_math_cos(val))
                sin_vals.append(_math_sin(val))
            cos_cached.append(tuple(cos_vals))
            sin_cached.append(tuple(sin_vals))
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def apply_rotary_pos_emb(self, x, pos):
        cos = self.cos_cached[pos]
        sin = self.sin_cached[pos]
        half = len(x) >> 1
        out = [0.0] * (half << 1)
        for i in range(half):
            idx = i << 1
            x1 = x[idx]
            x2 = x[idx + 1]
            c = cos[i]
            s = sin[i]
            out[idx] = x1 * c - x2 * s
            out[idx + 1] = x1 * s + x2 * c
        return out


class LlamaAttention:
    __slots__ = ('hidden_size', 'num_heads', 'head_dim', 'scale',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'rope',
                 '_qkv_weight', '_head_slices')

    def __init__(self, config: Dict, weights: Dict, prefix: str):
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.scale = 1.0 / _math_sqrt(self.head_dim)

        q_w = weights[f"{prefix}.q_proj.weight"]
        k_w = weights[f"{prefix}.k_proj.weight"]
        v_w = weights[f"{prefix}.v_proj.weight"]
        self.o_proj = Linear(weights[f"{prefix}.o_proj.weight"])

        # Fuse Q/K/V weights into a single matrix for one pass
        self._qkv_weight = q_w + k_w + v_w  # List concat: 3*hidden rows

        # Pre-compute head slice indices
        hd = self.head_dim
        hs = self.hidden_size
        self._head_slices = [(h * hd, h * hd + hd) for h in range(self.num_heads)]

        self.rope = LlamaRotaryEmbedding(self.head_dim, config["max_position_embeddings"], config["rope_parameters"]["rope_theta"])

        # Not used anymore but keep references for compatibility
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    def forward(self, x, pos, context_k, context_v):
        # Fused QKV projection: one mat_vec_mul instead of three
        qkv = [sum(map(_mul, row, x)) for row in self._qkv_weight]

        hs = self.hidden_size
        q = qkv[:hs]
        k = qkv[hs:hs + hs]
        v = qkv[hs + hs:]

        num_heads = self.num_heads
        head_dim = self.head_dim
        scale = self.scale
        rope_apply = self.rope.apply_rotary_pos_emb
        head_slices = self._head_slices

        q_heads = []
        k_heads = []
        v_heads = []

        for start, end in head_slices:
            q_h = q[start:end]
            k_h = k[start:end]

            q_h = rope_apply(q_h, pos)
            k_h = rope_apply(k_h, pos)

            q_heads.append(q_h)
            k_heads.append(k_h)
            v_heads.append(v[start:end])

        new_k_row = k_heads
        new_v_row = v_heads

        # Use append for KV cache instead of list concat (O(1) amortized vs O(n))
        all_k = context_k + [k_heads]
        all_v = context_v + [v_heads]

        seq_len = len(all_k)

        # Pre-extract per-head KV for faster inner loop
        concat_out = []
        for h in range(num_heads):
            q_h = q_heads[h]

            # Compute attention scores
            scores = []
            for t in range(seq_len):
                dot = sum(map(_mul, q_h, all_k[t][h]))
                scores.append(dot * scale)

            probs = softmax(scores)

            # Weighted sum of values
            out_h = [0.0] * head_dim
            for t in range(seq_len):
                v_t_h = all_v[t][h]
                prob = probs[t]
                for d in range(head_dim):
                    out_h[d] += v_t_h[d] * prob

            concat_out.extend(out_h)

        final_out = self.o_proj.forward(concat_out)

        return final_out, new_k_row, new_v_row


class LlamaMLP:
    __slots__ = ('_gate_weight', '_up_weight', '_down_weight')

    def __init__(self, config: Dict, weights: Dict, prefix: str):
        self._gate_weight = weights[f"{prefix}.gate_proj.weight"]
        self._up_weight = weights[f"{prefix}.up_proj.weight"]
        self._down_weight = weights[f"{prefix}.down_proj.weight"]

    def forward(self, x):
        # Inline mat_vec_mul to avoid function call overhead
        gate = [sum(map(_mul, row, x)) for row in self._gate_weight]
        up = [sum(map(_mul, row, x)) for row in self._up_weight]

        # Fuse silu + elem_mul in single pass (avoid intermediate list)
        inter = [silu(g) * u for g, u in zip(gate, up)]

        return [sum(map(_mul, row, inter)) for row in self._down_weight]


class LlamaDecoderLayer:
    __slots__ = ('input_layernorm', 'post_attention_layernorm', 'self_attn', 'mlp')

    def __init__(self, config: Dict, weights: Dict, layer_idx: int):
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = RMSNorm(weights[f"{prefix}.input_layernorm.weight"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(weights[f"{prefix}.post_attention_layernorm.weight"], config["rms_norm_eps"])
        self.self_attn = LlamaAttention(config, weights, f"{prefix}.self_attn")
        self.mlp = LlamaMLP(config, weights, f"{prefix}.mlp")

    def forward(self, x, pos, kv_cache):
        residual = x
        x_norm = self.input_layernorm.forward(x)

        context_k = kv_cache.get("k", [])
        context_v = kv_cache.get("v", [])

        attn_out, new_k, new_v = self.self_attn.forward(x_norm, pos, context_k, context_v)

        x = list(map(_add, residual, attn_out))

        kv_cache["k"] = context_k + [new_k]
        kv_cache["v"] = context_v + [new_v]

        residual = x
        x_norm = self.post_attention_layernorm.forward(x)
        mlp_out = self.mlp.forward(x_norm)

        x = list(map(_add, residual, mlp_out))

        return x, kv_cache


class TinyLlama:
    __slots__ = ('config', 'embed_tokens', 'norm', 'lm_head', 'layers',
                 '_num_layers', '_eos_token_id')

    def __init__(self, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.safetensors")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config
        weights = load_safetensors(weights_path)

        self.embed_tokens = weights["model.embed_tokens.weight"]
        self.norm = RMSNorm(weights["model.norm.weight"], config["rms_norm_eps"])
        self.lm_head = Linear(weights["lm_head.weight"])

        num_layers = config["num_hidden_layers"]
        self._num_layers = num_layers
        self.layers = [LlamaDecoderLayer(config, weights, i) for i in range(num_layers)]

        # Free raw weights dict after building layers
        del weights

    def forward(self, input_ids, start_pos=0, kv_caches=None):
        num_layers = self._num_layers
        if kv_caches is None:
            kv_caches = [{} for _ in range(num_layers)]

        hidden_states = [self.embed_tokens[idx] for idx in input_ids]

        seq_len = len(input_ids)
        layers = self.layers

        for i in range(num_layers):
            layer = layers[i]
            kv_i = kv_caches[i]
            new_hidden_states = []
            for t in range(seq_len):
                h, kv_i = layer.forward(hidden_states[t], start_pos + t, kv_i)
                new_hidden_states.append(h)
            kv_caches[i] = kv_i
            hidden_states = new_hidden_states

        last_hidden = self.norm.forward(hidden_states[-1])
        logits = self.lm_head.forward(last_hidden)

        return logits, kv_caches

    def generate(self, input_ids, max_new_tokens, eos_token_id):
        generated = []
        num_layers = self._num_layers
        kv_caches = [{} for _ in range(num_layers)]

        logits, kv_caches = self.forward(input_ids, start_pos=0, kv_caches=kv_caches)

        next_token = self._argmax(logits)
        generated.append(next_token)

        if next_token == eos_token_id:
            return generated

        cur_pos = len(input_ids)

        for _ in range(max_new_tokens - 1):
            logits, kv_caches = self.forward([next_token], start_pos=cur_pos, kv_caches=kv_caches)
            next_token = self._argmax(logits)
            generated.append(next_token)

            if next_token == eos_token_id:
                break
            cur_pos += 1

        return generated

    def _argmax(self, logits):
        best_idx = 0
        best_val = logits[0]
        for i in range(1, len(logits)):
            if logits[i] > best_val:
                best_val = logits[i]
                best_idx = i
        return best_idx
