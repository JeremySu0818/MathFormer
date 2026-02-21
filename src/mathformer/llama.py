import json
import math
import struct
import os
import operator

_add = operator.add
_sub = operator.sub
_mul = operator.mul
_exp = math.exp
_sqrt = math.sqrt
_cos = math.cos
_sin = math.sin


def vec_add(a, b):
    return list(map(_add, a, b))

def vec_sub(a, b):
    return list(map(_sub, a, b))

def vec_mul_scalar(v, s):
    return [x * s for x in v]

def vec_elem_mul(a, b):
    return list(map(_mul, a, b))

def mat_vec_mul(W, x):
    return [sum(map(_mul, row, x)) for row in W]

def softmax(x):
    m = max(x)
    e = [_exp(v - m) for v in x]
    inv = 1.0 / sum(e)
    return [v * inv for v in e]

def silu(x):
    return x / (1.0 + _exp(-x))

def rms_norm(x, w, eps):
    sc = 1.0 / _sqrt(sum(map(_mul, x, x)) / len(x) + eps)
    return [v * sc * wi for v, wi in zip(x, w)]


def load_safetensors(path):
    with open(path, "rb") as f:
        data = f.read()

    header_size = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8:8 + header_size])
    base = 8 + header_size

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        off = info["data_offsets"]
        start = base + off[0]
        end = base + off[1]
        shape = info["shape"]
        dtype = info["dtype"]

        if dtype == "F32":
            n = (end - start) // 4
            raw = struct.unpack_from(f"<{n}f", data, start)
        elif dtype in ("BF16", "F16"):
            raise NotImplementedError(f"{dtype} not supported")
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        if len(shape) == 1:
            tensors[name] = raw
        elif len(shape) == 2:
            rows, cols = shape
            tensors[name] = tuple(raw[r * cols:(r + 1) * cols] for r in range(rows))
        else:
            def reshape(it, dims):
                if len(dims) == 1:
                    return tuple(next(it) for _ in range(dims[0]))
                return tuple(reshape(it, dims[1:]) for _ in range(dims[0]))
            tensors[name] = reshape(iter(raw), shape)

    return tensors


class TinyLlama:
    """Fully-inlined Llama inference engine optimized for tiny models."""

    __slots__ = (
        'config', 'embed_tokens',
        '_norm_eps', '_lm_head_w',
        '_num_layers', '_hidden_size', '_num_heads', '_head_dim',
        '_attn_scale', '_intermediate_size',
        '_rope_cos', '_rope_sin',
        '_qkv_w', '_o_w', '_gate_up_w', '_down_w',
        '_use_hd4',
    )

    def __init__(self, model_path):
        config_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.safetensors")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.config = config
        W = load_safetensors(weights_path)

        self.embed_tokens = W["model.embed_tokens.weight"]
        self._norm_eps = config["rms_norm_eps"]
        
        nw = W["model.norm.weight"]
        self._lm_head_w = [[w * nw[j] for j, w in enumerate(row)] for row in W["lm_head.weight"]]

        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        hd = config["head_dim"]
        inter = config["intermediate_size"]
        n_layers = config["num_hidden_layers"]
        eps = config["rms_norm_eps"]
        rope_theta = config["rope_parameters"]["rope_theta"]
        max_pos = config["max_position_embeddings"]

        self._num_layers = n_layers
        self._hidden_size = hs
        self._num_heads = nh
        self._head_dim = hd
        self._attn_scale = 1.0 / _sqrt(hd)
        self._intermediate_size = inter
        self._use_hd4 = (hd == 4)

        inv_freq = [1.0 / (rope_theta ** (i / hd)) for i in range(0, hd, 2)]
        self._rope_cos = [tuple(_cos(p * f) for f in inv_freq) for p in range(max_pos)]
        self._rope_sin = [tuple(_sin(p * f) for f in inv_freq) for p in range(max_pos)]

        qkv_w, o_w, gate_up_w, down_w = [], [], [], []
        for i in range(n_layers):
            pfx = f"model.layers.{i}"
            
            raw_qkv = (
                W[f"{pfx}.self_attn.q_proj.weight"] +
                W[f"{pfx}.self_attn.k_proj.weight"] +
                W[f"{pfx}.self_attn.v_proj.weight"]
            )
            w1 = W[f"{pfx}.input_layernorm.weight"]
            qkv_w.append([[w * w1[j] for j, w in enumerate(row)] for row in raw_qkv])
            
            o_w.append(W[f"{pfx}.self_attn.o_proj.weight"])
            
            raw_gu = (
                W[f"{pfx}.mlp.gate_proj.weight"] +
                W[f"{pfx}.mlp.up_proj.weight"]
            )
            w2 = W[f"{pfx}.post_attention_layernorm.weight"]
            gate_up_w.append([[w * w2[j] for j, w in enumerate(row)] for row in raw_gu])
            
            down_w.append(W[f"{pfx}.mlp.down_proj.weight"])

        self._qkv_w = qkv_w
        self._o_w = o_w
        self._gate_up_w = gate_up_w
        self._down_w = down_w

        del W

    def forward(self, input_ids, start_pos=0, kv_caches=None):
        n_layers = self._num_layers
        if kv_caches is None:
            kv_caches = [{"k": [], "v": []} for _ in range(n_layers)]
        
        hidden = [self.embed_tokens[idx] for idx in input_ids]
        seq_len = len(input_ids)

        hs = self._hidden_size
        nh = self._num_heads
        hd = self._head_dim
        scale = self._attn_scale
        rope_cos = self._rope_cos
        rope_sin = self._rope_sin
        eps = self._norm_eps
        use_hd4 = self._use_hd4

        for li in range(n_layers):
            qkv_w = self._qkv_w[li]
            o_w = self._o_w[li]
            gu_w = self._gate_up_w[li]
            d_w = self._down_w[li]

            kv = kv_caches[li]
            if "k" not in kv:
                kv["k"] = []
                kv["v"] = []
            ck = kv["k"]
            cv = kv["v"]

            new_hidden = []
            for t in range(seq_len):
                x = hidden[t]
                pos = start_pos + t

                sc = 1.0 / _sqrt(sum(map(_mul, x, x)) / len(x) + eps)
                qkv = [sum(map(_mul, row, x)) * sc for row in qkv_w]
                q_all = qkv[:hs]
                k_all = qkv[hs:hs + hs]
                v_all = qkv[hs + hs:]

                cos_p = rope_cos[pos]
                sin_p = rope_sin[pos]

                if use_hd4:
                    c0, c1 = cos_p[0], cos_p[1]
                    s0, s1 = sin_p[0], sin_p[1]
                    q_heads = []
                    k_heads = []
                    v_heads = []

                    for h in range(nh):
                        si = h << 2
                        q0, q1, q2, q3 = q_all[si], q_all[si + 1], q_all[si + 2], q_all[si + 3]
                        k0, k1, k2, k3 = k_all[si], k_all[si + 1], k_all[si + 2], k_all[si + 3]
                        q_heads.append((
                            q0 * c0 - q2 * s0,
                            q1 * c1 - q3 * s1,
                            q2 * c0 + q0 * s0,
                            q3 * c1 + q1 * s1
                        ))
                        k_heads.append((
                            k0 * c0 - k2 * s0,
                            k1 * c1 - k3 * s1,
                            k2 * c0 + k0 * s0,
                            k3 * c1 + k1 * s1
                        ))
                        v_heads.append((v_all[si], v_all[si + 1], v_all[si + 2], v_all[si + 3]))

                    ck.append(k_heads)
                    cv.append(v_heads)
                    T = len(ck)

                    concat_out = []
                    for h in range(nh):
                        rq0, rq1, rq2, rq3 = q_heads[h]
                        scores = [0.0] * T
                        for tt in range(T):
                            kk = ck[tt][h]
                            scores[tt] = (rq0 * kk[0] + rq1 * kk[1] + rq2 * kk[2] + rq3 * kk[3]) * scale

                        sm = max(scores)
                        e = [_exp(v - sm) for v in scores]
                        inv = 1.0 / sum(e)

                        o0 = 0.0
                        o1 = 0.0
                        o2 = 0.0
                        o3 = 0.0
                        for tt in range(T):
                            p = e[tt] * inv
                            vv = cv[tt][h]
                            o0 += vv[0] * p
                            o1 += vv[1] * p
                            o2 += vv[2] * p
                            o3 += vv[3] * p
                        concat_out.append(o0)
                        concat_out.append(o1)
                        concat_out.append(o2)
                        concat_out.append(o3)
                else:
                    half_hd = hd >> 1
                    q_heads = []
                    k_heads = []
                    v_heads = []

                    for h in range(nh):
                        si = h * hd
                        rq = [0.0] * hd
                        rk = [0.0] * hd
                        for i in range(half_hd):
                            c = cos_p[i]
                            s = sin_p[i]
                            qi_first = q_all[si + i]
                            qi_second = q_all[si + i + half_hd]
                            ki_first = k_all[si + i]
                            ki_second = k_all[si + i + half_hd]
                            rq[i] = qi_first * c - qi_second * s
                            rq[i + half_hd] = qi_second * c + qi_first * s
                            rk[i] = ki_first * c - ki_second * s
                            rk[i + half_hd] = ki_second * c + ki_first * s
                        q_heads.append(rq)
                        k_heads.append(rk)
                        v_heads.append(v_all[si:si + hd])

                    ck.append(k_heads)
                    cv.append(v_heads)
                    T = len(ck)

                    concat_out = []
                    for h in range(nh):
                        q_h = q_heads[h]
                        scores = [sum(map(_mul, q_h, ck[tt][h])) * scale for tt in range(T)]

                        sm = max(scores)
                        e = [_exp(v - sm) for v in scores]
                        inv = 1.0 / sum(e)

                        out_h = [0.0] * hd
                        for tt in range(T):
                            vv = cv[tt][h]
                            p = e[tt] * inv
                            for d in range(hd):
                                out_h[d] += vv[d] * p
                        concat_out.extend(out_h)

                attn_out = [sum(map(_mul, row, concat_out)) for row in o_w]

                x = list(map(_add, x, attn_out))

                sc = 1.0 / _sqrt(sum(map(_mul, x, x)) / len(x) + eps)
                gu = [sum(map(_mul, row, x)) * sc for row in gu_w]
                mid = len(gu) >> 1
                inter = [silu(gu[i]) * gu[mid + i] for i in range(mid)]
                mlp_out = [sum(map(_mul, row, inter)) for row in d_w]

                x = list(map(_add, x, mlp_out))
                new_hidden.append(x)

            hidden = new_hidden

        x = hidden[-1]
        sc = 1.0 / _sqrt(sum(map(_mul, x, x)) / len(x) + eps)
        logits = [sum(map(_mul, row, x)) * sc for row in self._lm_head_w]

        return logits, kv_caches

    def generate(self, input_ids, max_new_tokens, eos_token_id):
        generated = []
        kv_caches = [{"k": [], "v": []} for _ in range(self._num_layers)]

        logits, kv_caches = self.forward(input_ids, start_pos=0, kv_caches=kv_caches)
        next_token = _argmax(logits)
        generated.append(next_token)
        if next_token == eos_token_id:
            return generated

        cur_pos = len(input_ids)
        for _ in range(max_new_tokens - 1):
            logits, kv_caches = self.forward([next_token], start_pos=cur_pos, kv_caches=kv_caches)
            next_token = _argmax(logits)
            generated.append(next_token)
            if next_token == eos_token_id:
                break
            cur_pos += 1

        return generated


def _argmax(logits):
    best_idx = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best_idx = i
    return best_idx
