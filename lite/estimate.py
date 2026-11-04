"""
Estimate parameter counts, weight memory and KV-cache memory for a config.
Pure arithmetic: no torch, no weights needed.

    python lite/estimate.py inference/configs/config_16B.json --seq-len 4096
"""
import json
import sys
from argparse import ArgumentParser


def count_params(c: dict) -> dict:
    d, H = c["dim"], c["n_heads"]
    dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    qr, kvr = c.get("q_lora_rank", 0), c["kv_lora_rank"]
    E, k, mi = c["n_routed_experts"], c["n_activated_experts"], c["moe_inter_dim"]
    attn = (d * H * (dn + dr) if qr == 0 else d * qr + qr + qr * H * (dn + dr)) \
        + d * (kvr + dr) + kvr + kvr * H * (dn + dv) + H * dv * d
    dense_ffn = 3 * d * c["inter_dim"]
    expert = 3 * d * mi
    shared = 3 * d * c.get("n_shared_experts", 0) * mi
    router = E * d + (E if c.get("gate_bias", c.get("score_func") == "sigmoid") else 0)
    nd, nl = c["n_dense_layers"], c["n_layers"]
    embed = head = c["vocab_size"] * d
    norms = nl * 2 * d + d
    total_layers = nl * attn + nd * dense_ffn + (nl - nd) * (E * expert + shared + router)
    active_layers = nl * attn + nd * dense_ffn + (nl - nd) * (k * expert + shared + router)
    return {
        "total": embed + head + norms + total_layers,
        "active_per_token": embed // c["vocab_size"] + head + norms + active_layers,
        "embed": embed, "head": head,
        "linear": nl * (attn - kvr - (qr if qr else 0)) + nd * dense_ffn + (nl - nd) * (E * expert + shared),
        "routed_expert_share": (nl - nd) * E * expert,
    }


def weight_bytes(p: dict, quant: str, group: int = 128) -> float:
    """Approximate on-disk/in-RAM size; head and embedding use int8 when quant != none."""
    per = {"none": 2.0, "int8": 1.0, "int4": 0.5 + 4.0 / group}[quant]
    side = 2.0 if quant == "none" else 1.0
    other = p["total"] - p["linear"] - p["embed"] - p["head"]
    return p["linear"] * per + (p["embed"] + p["head"]) * side + other * 2.0


def kv_cache_bytes(c: dict, seq_len: int, batch: int = 1, elem: int = 2) -> int:
    return c["n_layers"] * batch * seq_len * (c["kv_lora_rank"] + c["qk_rope_head_dim"]) * elem


def main(argv=None):
    ap = ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seq-len", type=int, default=4096)
    a = ap.parse_args(argv)
    for path in a.configs:
        with open(path) as f:
            c = json.load(f)
        p = count_params(c)
        print(f"{path}")
        print(f"  params: {p['total'] / 1e9:.2f} B total, {p['active_per_token'] / 1e9:.2f} B active/token "
              f"({100 * p['routed_expert_share'] / p['total']:.0f}% in routed experts)")
        for q in ("none", "int8", "int4"):
            print(f"  weights {q:>4}: {weight_bytes(p, q) / 2**30:8.1f} GiB")
        print(f"  KV cache @ {a.seq_len} tokens (bf16, batch 1): {kv_cache_bytes(c, a.seq_len) / 2**20:.0f} MiB"
              f"  | reference demo default 8 x 16384: {kv_cache_bytes(c, 16384, 8) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main(sys.argv[1:])
