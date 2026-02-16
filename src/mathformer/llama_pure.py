import json
import math
import struct
import os
from typing import List, Dict, Tuple, Optional, Any, Union

def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def vec_mul_scalar(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]

def vec_elem_mul(a: List[float], b: List[float]) -> List[float]:
    return [x * y for x, y in zip(a, b)]

def mat_vec_mul(W: List[List[float]], x: List[float]) -> List[float]:
    rows = len(W)
    cols = len(W[0])
    assert len(x) == cols, f"Shape mismatch: W={rows}x{cols}, x={len(x)}"
    
    result = []
    for i in range(rows):
        s = 0.0
        row = W[i]
        for j in range(cols):
            s += row[j] * x[j]
        result.append(s)
    return result

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    M = len(A)
    K = len(A[0])
    rows_B = len(B)
    N = len(B[0])
    assert K == rows_B
    
    B_T = [[B[j][i] for j in range(K)] for i in range(N)]
    
    result = []
    for i in range(M):
        row_res = []
        row_A = A[i]
        for j in range(N):
            col_B = B_T[j]
            s = 0.0
            for k in range(K):
                s += row_A[k] * col_B[k]
            row_res.append(s)
        result.append(row_res)
    return result

def softmax(x: List[float]) -> List[float]:
    max_val = max(x)
    exps = [math.exp(val - max_val) for val in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))

def rms_norm(x: List[float], w: List[float], eps: float) -> List[float]:
    sum_sq = sum(v * v for v in x)
    mean_sq = sum_sq / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [val * scale * weight for val, weight in zip(x, w)]


def load_safetensors(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        
        header_json_bytes = f.read(header_size)
        header = json.loads(header_json_bytes.decode("utf-8"))
        
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            
            data_offsets = info["data_offsets"]
            start = data_offsets[0] + 8 + header_size
            end = data_offsets[1] + 8 + header_size
            
            current_pos = f.tell()
            f.seek(start)
            data_bytes = f.read(end - start)
            f.seek(current_pos)
            
            shape = info["shape"]
            dtype = info["dtype"]
            
            if dtype == "F32":
                num_elements = len(data_bytes) // 4
                raw_data = struct.unpack(f"<{num_elements}f", data_bytes)
            elif dtype == "BF16" or dtype == "F16":
                raise NotImplementedError(f"Dtype {dtype} not implemented in pure python reader yet")
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
                
            def reshape(data_iter, dims):
                if len(dims) == 1:
                    return [next(data_iter) for _ in range(dims[0])]
                return [reshape(data_iter, dims[1:]) for _ in range(dims[0])]
            
            data_iter = iter(raw_data)
            tensors[name] = reshape(data_iter, shape)
            
        return tensors

class Linear:
    def __init__(self, weight: List[List[float]], bias: Optional[List[float]] = None):
        self.weight = weight
        self.bias = bias
        
    def forward(self, x: List[float]) -> List[float]:
        out = mat_vec_mul(self.weight, x)
        if self.bias:
            out = vec_add(out, self.bias)
        return out

class RMSNorm:
    def __init__(self, weight: List[float], eps: float = 1e-6):
        self.weight = weight
        self.eps = eps
        
    def forward(self, x: List[float]) -> List[float]:
        return rms_norm(x, self.weight, self.eps)

class LlamaRotaryEmbedding:
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.base = base
        self.inv_freq = [1.0 / (base ** (i / dim)) for i in range(0, dim, 2)]
        self.cos_cached = []
        self.sin_cached = []
        
        for pos in range(max_position_embeddings):
            cos_vals = []
            sin_vals = []
            for freq in self.inv_freq:
                val = pos * freq
                cos_vals.append(math.cos(val))
                sin_vals.append(math.sin(val))
            self.cos_cached.append(cos_vals)
            self.sin_cached.append(sin_vals)

    def apply_rotary_pos_emb(self, x: List[float], pos: int) -> List[float]:
        out = [0.0] * len(x)
        cos = self.cos_cached[pos]
        sin = self.sin_cached[pos]
        
        for i in range(len(x) // 2):
            x1 = x[2*i]
            x2 = x[2*i + 1]
            c = cos[i]
            s = sin[i]
            out[2*i] = x1 * c - x2 * s
            out[2*i + 1] = x1 * s + x2 * c
        return out

class LlamaAttention:
    def __init__(self, config: Dict, weights: Dict, prefix: str):
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = Linear(weights[f"{prefix}.q_proj.weight"])
        self.k_proj = Linear(weights[f"{prefix}.k_proj.weight"])
        self.v_proj = Linear(weights[f"{prefix}.v_proj.weight"])
        self.o_proj = Linear(weights[f"{prefix}.o_proj.weight"])
        
        self.rope = LlamaRotaryEmbedding(self.head_dim, config["max_position_embeddings"], config["rope_parameters"]["rope_theta"])

    def forward(self, x: List[float], pos: int, context_k: List[List[float]], context_v: List[List[float]]) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        q_heads = []
        k_heads = []
        v_heads = []
        
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            q_h = q[start:end]
            k_h = k[start:end]
            v_h = v[start:end]
            
            q_h = self.rope.apply_rotary_pos_emb(q_h, pos)
            k_h = self.rope.apply_rotary_pos_emb(k_h, pos)
            
            q_heads.append(q_h)
            k_heads.append(k_h)
            v_heads.append(v_h)
        
        new_k_row = k_heads
        new_v_row = v_heads
        
        all_k = context_k + [k_heads]
        all_v = context_v + [v_heads]
        
        seq_len = len(all_k)
        
        output_heads = []
        
        for h in range(self.num_heads):
            q_h = q_heads[h]
            
            scores = []
            for t in range(seq_len):
                k_t_h = all_k[t][h]
                dot = sum(a * b for a, b in zip(q_h, k_t_h))
                scores.append(dot * self.scale)
            
            probs = softmax(scores)
            
            out_h = [0.0] * self.head_dim
            for t in range(seq_len):
                v_t_h = all_v[t][h]
                prob = probs[t]
                for d in range(self.head_dim):
                    out_h[d] += v_t_h[d] * prob
            
            output_heads.append(out_h)
            
        concat_out = []
        for h in range(self.num_heads):
            concat_out.extend(output_heads[h])
            
        final_out = self.o_proj.forward(concat_out)
        
        return final_out, new_k_row, new_v_row

class LlamaMLP:
    def __init__(self, config: Dict, weights: Dict, prefix: str):
        self.gate_proj = Linear(weights[f"{prefix}.gate_proj.weight"])
        self.up_proj = Linear(weights[f"{prefix}.up_proj.weight"])
        self.down_proj = Linear(weights[f"{prefix}.down_proj.weight"])
        
    def forward(self, x: List[float]) -> List[float]:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        
        gate = [silu(val) for val in gate]
        inter = vec_elem_mul(gate, up)
        
        return self.down_proj.forward(inter)

class LlamaDecoderLayer:
    def __init__(self, config: Dict, weights: Dict, layer_idx: int):
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = RMSNorm(weights[f"{prefix}.input_layernorm.weight"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(weights[f"{prefix}.post_attention_layernorm.weight"], config["rms_norm_eps"])
        self.self_attn = LlamaAttention(config, weights, f"{prefix}.self_attn")
        self.mlp = LlamaMLP(config, weights, f"{prefix}.mlp")
        
    def forward(self, x: List[float], pos: int, kv_cache: Dict) -> Tuple[List[float], Dict]:
        residual = x
        x_norm = self.input_layernorm.forward(x)
        
        context_k = kv_cache.get("k", [])
        context_v = kv_cache.get("v", [])
        
        attn_out, new_k, new_v = self.self_attn.forward(x_norm, pos, context_k, context_v)
        
        x = vec_add(residual, attn_out)
        
        kv_cache["k"] = context_k + [new_k]
        kv_cache["v"] = context_v + [new_v]
        
        residual = x
        x_norm = self.post_attention_layernorm.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        
        x = vec_add(residual, mlp_out)
        
        return x, kv_cache

class TinyLlama:
    def __init__(self, model_path: str):
        self.config_path = os.path.join(model_path, "config.json")
        self.weights_path = os.path.join(model_path, "model.safetensors")
        
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            
        self.weights = load_safetensors(self.weights_path)
        
        self.embed_tokens = self.weights["model.embed_tokens.weight"]
        self.norm = RMSNorm(self.weights["model.norm.weight"], self.config["rms_norm_eps"])
        self.lm_head = Linear(self.weights["lm_head.weight"])
        
        self.layers = []
        for i in range(self.config["num_hidden_layers"]):
            self.layers.append(LlamaDecoderLayer(self.config, self.weights, i))
            
    def forward(self, input_ids: List[int], start_pos: int = 0, kv_caches: Optional[List[Dict]] = None) -> Tuple[List[float], List[Dict]]:
        
        if kv_caches is None:
            kv_caches = [{} for _ in range(len(self.layers))]
            
        hidden_states = [self.embed_tokens[idx] for idx in input_ids]
        
        seq_len = len(input_ids)
        
        for i, layer in enumerate(self.layers):
            new_hidden_states = []
            for t in range(seq_len):
                h, kv_caches[i] = layer.forward(hidden_states[t], start_pos + t, kv_caches[i])
                new_hidden_states.append(h)
            hidden_states = new_hidden_states
            
        hidden_states = [self.norm.forward(h) for h in hidden_states]
        
        last_hidden = hidden_states[-1]
        logits = self.lm_head.forward(last_hidden)
        
        return logits, kv_caches

    def generate(self, input_ids: List[int], max_new_tokens: int, eos_token_id: int) -> List[int]:
        generated = []
        current_ids = input_ids[:]
        
        kv_caches = [{} for _ in range(len(self.layers))]
        
        logits, kv_caches = self.forward(current_ids, start_pos=0, kv_caches=kv_caches)
        
        next_token = self._argmax(logits)
        generated.append(next_token)
        
        if next_token == eos_token_id:
            return generated
            
        cur_pos = len(current_ids)
        
        for _ in range(max_new_tokens - 1):
             logits, kv_caches = self.forward([next_token], start_pos=cur_pos, kv_caches=kv_caches)
             next_token = self._argmax(logits)
             generated.append(next_token)
             
             if next_token == eos_token_id:
                 break
             cur_pos += 1
             
        return generated

    def _argmax(self, logits: List[float]) -> int:
        best_idx = 0
        best_val = logits[0]
        for i in range(1, len(logits)):
            if logits[i] > best_val:
                best_val = logits[i]
                best_idx = i
        return best_idx
