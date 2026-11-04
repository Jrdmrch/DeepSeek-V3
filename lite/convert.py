"""
Convert a Hugging Face DeepSeek-V2 / V2-Lite / V3 checkpoint into a compact,
single-device DeepSeek-Lite checkpoint.

* Streams one tensor at a time (memory use ~ one output shard, default 2 GB).
* Dequantizes the official FP8 weights with pure PyTorch (no GPU or Triton needed).
* Optionally quantizes linear weights to int8 or int4 (weight-only, round-to-nearest).
* Drops the multi-token-prediction (MTP) layer, which plain generation does not use.
* Writes config.json for lite/model.py and copies the tokenizer files.

Example:
    python lite/convert.py --hf-ckpt-path ./DeepSeek-V2-Lite-Chat --save-path ./v2-lite-int4 --quant int4
"""
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from glob import glob

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import fp8_block_dequant, quantize_int4, quantize_int8  # noqa: E402

RENAME = {
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate": "gate",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "norm": "norm",
    "lm_head": "head",
}


def hf_config_to_args(cfg: dict) -> dict:
    """Map a Hugging Face DeepSeek config.json onto lite ModelArgs fields."""
    rs = cfg.get("rope_scaling") or {}
    yarn = rs.get("type", rs.get("rope_type")) == "yarn"
    args = {
        "vocab_size": cfg["vocab_size"],
        "dim": cfg["hidden_size"],
        "inter_dim": cfg["intermediate_size"],
        "moe_inter_dim": cfg["moe_intermediate_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_dense_layers": cfg.get("first_k_dense_replace", 0),
        "n_heads": cfg["num_attention_heads"],
        "n_routed_experts": cfg["n_routed_experts"],
        "n_shared_experts": cfg.get("n_shared_experts") or 0,
        "n_activated_experts": cfg["num_experts_per_tok"],
        "n_expert_groups": cfg.get("n_group") or 1,
        "n_limited_groups": cfg.get("topk_group") or 1,
        "score_func": cfg.get("scoring_func", "softmax"),
        "route_scale": float(cfg.get("routed_scaling_factor", 1.0)),
        "gate_bias": cfg.get("topk_method") == "noaux_tc",
        "q_lora_rank": cfg.get("q_lora_rank") or 0,
        "kv_lora_rank": cfg["kv_lora_rank"],
        "qk_nope_head_dim": cfg["qk_nope_head_dim"],
        "qk_rope_head_dim": cfg["qk_rope_head_dim"],
        "v_head_dim": cfg["v_head_dim"],
        "norm_eps": cfg.get("rms_norm_eps", 1e-6),
        "rope_theta": float(cfg.get("rope_theta", 10000.0)),
        "yarn": yarn,
    }
    if yarn:
        args.update(
            original_seq_len=rs.get("original_max_position_embeddings", 4096),
            rope_factor=float(rs.get("factor", 1.0)),
            beta_fast=rs.get("beta_fast", 32),
            beta_slow=rs.get("beta_slow", 1),
            mscale=float(rs.get("mscale_all_dim", 1.0)),
        )
    if cfg.get("moe_layer_freq", 1) != 1:
        raise NotImplementedError("moe_layer_freq != 1 is not supported")
    if cfg.get("topk_method") == "greedy" and args["n_expert_groups"] > 1:
        raise NotImplementedError("greedy routing with expert groups is not supported")
    return args


def rename(hf_name: str) -> str:
    name = hf_name[len("model."):] if hf_name.startswith("model.") else hf_name
    parts = name.replace("self_attn", "attn").replace("mlp", "ffn").split(".")
    if parts[-1] == "e_score_correction_bias":
        parts[-1] = "bias"
    key = parts[-2]
    if key not in RENAME:
        raise KeyError(f"Unknown tensor '{hf_name}'")
    parts[-2] = RENAME[key]
    return ".".join(parts)


def is_quantizable(name: str) -> bool:
    """Linear weights other than the router (small and precision-sensitive) and the embedding."""
    return name.endswith(".weight") and not name.endswith("ffn.gate.weight") and "norm" not in name.split(".")[-2] \
        and not name.startswith("embed.")


def encode(name: str, w: torch.Tensor, quant: str, head_quant: str, embed_quant: str, group_size: int, dtype):
    """Return {name: tensor} entries for one (already dequantized) weight."""
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    mode = "none"
    if name == "head.weight":
        mode = head_quant
    elif name == "embed.weight":
        mode = embed_quant
    elif w.dim() == 2 and is_quantizable(name):
        mode = quant
    if mode == "int4" and w.size(1) % group_size:
        mode = "int8"  # fall back if the input dim does not split into groups
    if mode == "int8":
        q, s = quantize_int8(w)
        return {name: q, f"{base}.scale": s}
    if mode == "int4":
        q, s = quantize_int4(w, group_size)
        return {name: q, f"{base}.scale": s}
    if name.endswith(".bias") or "norm" in name:
        return {name: w.float()}
    return {name: w.to(dtype).contiguous()}


def main(hf_ckpt_path, save_path, quant="int8", head_quant=None, embed_quant=None, group_size=128,
         dtype="bf16", shard_gb=2.0):
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    head_quant = head_quant or ("int8" if quant != "none" else "none")
    embed_quant = embed_quant or ("int8" if quant != "none" else "none")

    with open(os.path.join(hf_ckpt_path, "config.json")) as f:
        hf_cfg = json.load(f)
    args = hf_config_to_args(hf_cfg)
    n_layers = args["n_layers"]

    files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {hf_ckpt_path}")
    handles = {f: safe_open(f, framework="pt", device="cpu") for f in files}
    location = {k: f for f, h in handles.items() for k in h.keys()}

    os.makedirs(save_path, exist_ok=True)
    shard, shard_bytes, shard_idx, total_bytes, skipped = {}, 0, 0, 0, 0

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if shard:
            save_file(shard, os.path.join(save_path, f"model-{shard_idx:05d}.safetensors"))
            shard_idx += 1
            shard, shard_bytes = {}, 0

    names = sorted(k for k in location if not k.endswith("_scale_inv"))
    for i, hf_name in enumerate(names):
        parts = hf_name.split(".")
        if "layers" in parts and int(parts[parts.index("layers") + 1]) >= n_layers:
            skipped += 1  # MTP module
            continue
        if "rotary_emb" in hf_name:
            continue
        w = handles[location[hf_name]].get_tensor(hf_name)
        if w.element_size() == 1 and w.is_floating_point():
            s_name = hf_name + "_scale_inv"
            if s_name not in location:
                raise KeyError(f"FP8 tensor {hf_name} has no {s_name}")
            w = fp8_block_dequant(w, handles[location[s_name]].get_tensor(s_name))
        for k, t in encode(rename(hf_name), w, quant, head_quant, embed_quant, group_size, torch_dtype).items():
            shard[k] = t
            nbytes = t.numel() * t.element_size()
            shard_bytes += nbytes
            total_bytes += nbytes
        if shard_bytes >= shard_gb * 1e9:
            flush()
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(names)} tensors, {total_bytes / 1e9:.2f} GB written")
    flush()

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(args, f, indent=2)
    for p in glob(os.path.join(hf_ckpt_path, "*token*")):
        shutil.copyfile(p, os.path.join(save_path, os.path.basename(p)))
    print(f"Done: {total_bytes / 1e9:.2f} GB in {shard_idx} shard(s) -> {save_path}"
          f" (quant={quant}, head={head_quant}, embed={embed_quant}; skipped {skipped} MTP tensors)")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hf-ckpt-path", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--quant", choices=["none", "int8", "int4"], default="int8",
                   help="format for linear weights (default int8)")
    p.add_argument("--head-quant", choices=["none", "int8", "int4"], default=None,
                   help="format for the output head (default: int8 unless --quant none)")
    p.add_argument("--embed-quant", choices=["none", "int8"], default=None,
                   help="format for the embedding (default: int8 unless --quant none)")
    p.add_argument("--group-size", type=int, default=128, help="int4 group size")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="dtype for unquantized weights")
    p.add_argument("--shard-gb", type=float, default=2.0)
    a = p.parse_args()
    main(a.hf_ckpt_path, a.save_path, a.quant, a.head_quant, a.embed_quant, a.group_size, a.dtype, a.shard_gb)
