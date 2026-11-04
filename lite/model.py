"""
DeepSeek-Lite: a single-device, dependency-light re-implementation of the
DeepSeek-V2/V3 inference model in ``inference/model.py``.

Differences from the reference implementation (numerics are kept identical
for unquantized weights; see ``lite/tests``):

* Pure PyTorch: no Triton kernels, no ``torch.distributed``/NCCL. Runs on
  CPU, CUDA, or Apple MPS.
* Weight-only quantization: every linear layer accepts dense (bf16/fp16/fp32),
  int8 (per-output-channel), int4 (group-wise, packed) or block-FP8 weights.
  The storage format is detected from the checkpoint tensors at load time.
* KV cache is sized to what you ask for (default batch 1, 4096 tokens) instead
  of a fixed 8 x 16384 allocation.
* YaRN rope scaling depends on the model config, not on the cache length, so
  choosing a smaller context window does not silently change the math.
* Rotary embeddings use real arithmetic (no complex tensors) for MPS support.
* Causal mask is correct for multi-token forwards at ``start_pos > 0``, which
  enables KV-cache reuse across chat turns.
* Router bias is a config flag instead of the ``dim == 7168`` special case.
* Weights are loaded onto a meta-initialised model with ``assign`` semantics,
  so peak RAM during loading is ~1x the checkpoint size.
"""
import glob
import json
import math
import os
from dataclasses import dataclass, fields
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

FP8_BLOCK = 128


@dataclass
class ModelArgs:
    # runtime
    max_batch_size: int = 1
    max_seq_len: int = 4096
    # architecture
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    gate_bias: Optional[bool] = None  # None -> True iff score_func == "sigmoid" (V3 style)
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    norm_eps: float = 1e-6
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    yarn: bool = True  # apply YaRN scaling (all released DeepSeek-V2/V3 configs use it)
    # ignored keys kept for compatibility with inference/configs/*.json
    dtype: Optional[str] = None
    scale_fmt: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelArgs":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        args = cls(**d)
        if args.gate_bias is None:
            args.gate_bias = args.score_func == "sigmoid"
        return args

    @classmethod
    def from_json(cls, path: str, **overrides) -> "ModelArgs":
        with open(path) as f:
            d = json.load(f)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(d)


# --------------------------------------------------------------------------
# Quantization helpers (pure torch)
# --------------------------------------------------------------------------

def fp8_block_dequant(w: torch.Tensor, scale: torch.Tensor, dtype=torch.float32, block: int = FP8_BLOCK) -> torch.Tensor:
    """Dequantize a block-scaled FP8 weight (DeepSeek-V3 format) without Triton."""
    M, N = w.shape
    s = scale.float().repeat_interleave(block, 0)[:M].repeat_interleave(block, 1)[:, :N]
    return (w.float() * s).to(dtype)


def quantize_int8(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8. Returns (int8 [out,in], fp32 scale [out])."""
    w = w.float()
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def quantize_int4(w: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group-wise int4. Returns (uint8 packed [out,in/2], fp32 scale [out,in/group])."""
    out_f, in_f = w.shape
    if in_f % group_size or group_size % 2:
        raise ValueError(f"in_features={in_f} must be divisible by an even group_size={group_size}")
    g = w.float().view(out_f, in_f // group_size, group_size)
    scale = (g.abs().amax(dim=-1) / 7.0).clamp(min=1e-12)
    q = torch.round(g / scale[..., None]).clamp_(-8, 7).to(torch.int16) + 8  # 0..15
    q = q.view(out_f, in_f // 2, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
    return packed, scale


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, dtype) -> torch.Tensor:
    out_f, half = packed.shape
    lo = (packed & 0xF).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    q = torch.stack((lo, hi), dim=-1).view(out_f, half * 2)
    group = q.size(1) // scale.size(1)
    return (q.view(out_f, scale.size(1), group).to(dtype) * scale.to(dtype)[..., None]).view(out_f, -1)


class QuantLinear(nn.Module):
    """Bias-free linear layer whose storage format is decided at load time.

    format "dense": weight in compute dtype
    format "int8" : weight int8 [out,in], scale fp32 [out]
    format "int4" : weight uint8 [out,in/2], scale fp32 [out,in/group]
    format "fp8"  : weight float8_e4m3fn [out,in], scale fp32 [ceil(out/128), ceil(in/128)]
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = "dense"
        self.register_buffer("weight", torch.empty(out_features, in_features, device="meta"))
        self.register_buffer("scale", None)

    def set_weights(self, weight: torch.Tensor, scale: Optional[torch.Tensor], compute_dtype, device):
        if weight.dtype == torch.int8:
            self.format = "int8"
            exp = (self.out_features, self.in_features)
        elif weight.dtype == torch.uint8:
            self.format = "int4"
            exp = (self.out_features, self.in_features // 2)
        elif weight.element_size() == 1:  # float8
            self.format = "fp8"
            exp = (self.out_features, self.in_features)
        else:
            self.format = "dense"
            exp = (self.out_features, self.in_features)
            weight = weight.to(compute_dtype)
        if tuple(weight.shape) != exp:
            raise ValueError(f"weight shape {tuple(weight.shape)} != expected {exp} for format {self.format}")
        if self.format != "dense" and scale is None:
            raise ValueError(f"{self.format} weight is missing its scale tensor")
        self.weight = weight.to(device)
        self.scale = None if scale is None or self.format == "dense" else scale.float().to(device)

    def dense_weight(self, dtype) -> torch.Tensor:
        if self.format == "dense":
            return self.weight.to(dtype)
        if self.format == "int8":
            return self.weight.to(dtype) * self.scale.to(dtype)[:, None]
        if self.format == "int4":
            return unpack_int4(self.weight, self.scale, dtype)
        return fp8_block_dequant(self.weight, self.scale, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.format == "dense":
            return F.linear(x, self.weight)
        if self.format == "int8":  # scale is per output row, so apply it after the matmul
            return F.linear(x, self.weight.to(x.dtype)) * self.scale.to(x.dtype)
        return F.linear(x, self.dense_weight(x.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, format={self.format}"


class Embedding(nn.Module):
    """Embedding table, optionally int8 with one scale per row."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.register_buffer("weight", torch.empty(vocab_size, dim, device="meta"))
        self.register_buffer("scale", None)

    def set_weights(self, weight, scale, compute_dtype, device):
        if weight.dtype != torch.int8:
            weight, scale = weight.to(compute_dtype), None
        self.weight = weight.to(device)
        self.scale = None if scale is None else scale.float().to(device)

    def forward(self, tokens: torch.Tensor, dtype) -> torch.Tensor:
        e = F.embedding(tokens, self.weight)
        if self.scale is None:
            return e.to(dtype)
        return e.to(dtype) * self.scale[tokens].to(dtype)[..., None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device="meta"), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


# --------------------------------------------------------------------------
# Rotary embedding (YaRN), real-valued
# --------------------------------------------------------------------------

def precompute_rope(args: ModelArgs) -> Tuple[torch.Tensor, torch.Tensor]:
    dim, base, factor = args.qk_rope_head_dim, args.rope_theta, args.rope_factor

    def correction_dim(num_rot):
        return dim * math.log(args.original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if args.yarn and factor > 1:
        low = max(math.floor(correction_dim(args.beta_fast)), 0)
        high = min(math.ceil(correction_dim(args.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(args.max_seq_len, dtype=torch.float32), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [b, s, h, d] with interleaved (even, odd) pairs; cos/sin: [s, d/2]."""
    dtype = x.dtype
    xf = x.float().unflatten(-1, (-1, 2))
    x0, x1 = xf[..., 0], xf[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)
    return out.to(dtype)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class MLA(nn.Module):
    """Multi-head Latent Attention, weight-absorbed form (caches only the 576-d latent)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        if self.q_lora_rank == 0:
            self.wq = QuantLinear(args.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = QuantLinear(args.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
            self.wq_b = QuantLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = QuantLinear(args.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps)
        self.wkv_b = QuantLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = QuantLinear(self.n_heads * self.v_head_dim, args.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.yarn and args.rope_factor > 1:
            m = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= m * m
        self.register_buffer("kv_cache", None, persistent=False)
        self.register_buffer("pe_cache", None, persistent=False)

    def allocate_cache(self, args: ModelArgs, dtype, device):
        self.kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, dtype=dtype, device=device)
        self.pe_cache = torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim, dtype=dtype, device=device)

    def forward(self, x, start_pos: int, cos, sin, mask):
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        q = self.wq(x) if self.q_lora_rank == 0 else self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)
        kv, k_pe = torch.split(self.wkv_a(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin)

        wkv_b = self.wkv_b.dense_weight(x.dtype).view(self.n_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        kv_c, pe_c = self.kv_cache[:bsz, :end_pos], self.pe_cache[:bsz, :end_pos]
        scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_c) +
                  torch.einsum("bshr,btr->bsht", q_pe, pe_c)) * self.softmax_scale
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        out = torch.einsum("bsht,btc->bshc", scores, kv_c)
        out = torch.einsum("bshc,hdc->bshd", out, wkv_b[:, -self.v_head_dim:])
        return self.wo(out.flatten(2))


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = QuantLinear(dim, inter_dim)
        self.w2 = QuantLinear(inter_dim, dim)
        self.w3 = QuantLinear(dim, inter_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, device="meta"), requires_grad=False)
        if args.gate_bias:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, device="meta"), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        scores = F.linear(x, self.weight.to(x.dtype))
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_func == "softmax" else scores.float().sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias.float()
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1) if self.bias is None else scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=torch.bool).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(MLP(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts))
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i, c in enumerate(counts):
            if c == 0:
                continue  # only the experts that were routed to do any work
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
        return (y + self.shared_experts(x)).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, start_pos, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), start_pos, cos, sin, mask)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.embed = Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(Block(i, args) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = QuantLinear(args.dim, args.vocab_size)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.dtype = torch.float32

    def setup_runtime(self, dtype, device):
        """Allocate KV caches and rope tables (call after weights are loaded)."""
        self.dtype = dtype
        cos, sin = precompute_rope(self.args)
        self.rope_cos, self.rope_sin = cos.to(device), sin.to(device)
        for layer in self.layers:
            layer.attn.allocate_cache(self.args, dtype, device)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Returns float32 logits for the last position, shape [batch, vocab]."""
        bsz, seqlen = tokens.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(f"sequence end {end_pos} exceeds max_seq_len={self.max_seq_len}")
        if bsz > self.args.max_batch_size:
            raise ValueError(f"batch {bsz} exceeds max_batch_size={self.args.max_batch_size}")
        h = self.embed(tokens, self.dtype)
        cos, sin = self.rope_cos[start_pos:end_pos], self.rope_sin[start_pos:end_pos]
        mask = None
        if seqlen > 1:
            # query i (absolute position start_pos+i) may attend to keys 0..start_pos+i
            mask = torch.full((seqlen, end_pos), float("-inf"), device=tokens.device).triu_(start_pos + 1)
        for layer in self.layers:
            h = layer(h, start_pos, cos, sin, mask)
        h = self.norm(h)[:, -1]
        return self.head(h).float()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str):
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.split(":")[0], torch.bfloat16)


def load_state_dict_dir(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {path}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def build_model(args: ModelArgs, state_dict: Dict[str, torch.Tensor], device="cpu", dtype=torch.bfloat16) -> Transformer:
    """Create the model on the meta device and move checkpoint tensors straight into it."""
    with torch.device("meta"):
        model = Transformer(args)
    used = set()

    def take(name, required=True):
        if name not in state_dict:
            if required:
                raise KeyError(f"checkpoint is missing tensor '{name}'")
            return None
        used.add(name)
        return state_dict[name]

    for mname, mod in model.named_modules():
        prefix = f"{mname}." if mname else ""
        if isinstance(mod, (QuantLinear, Embedding)):
            mod.set_weights(take(prefix + "weight"), take(prefix + "scale", required=False), dtype, device)
        elif isinstance(mod, RMSNorm):
            mod.weight = nn.Parameter(take(prefix + "weight").float().to(device), requires_grad=False)
        elif isinstance(mod, Gate):
            mod.weight = nn.Parameter(take(prefix + "weight").to(dtype).to(device), requires_grad=False)
            if mod.bias is not None:
                mod.bias = nn.Parameter(take(prefix + "bias").float().to(device), requires_grad=False)
    unused = set(state_dict) - used
    if unused:
        raise KeyError(f"checkpoint has {len(unused)} unexpected tensors, e.g. {sorted(unused)[:5]}")
    model.setup_runtime(dtype, device)
    return model.eval()


def load_model(ckpt_path: str, device: Optional[str] = None, dtype=None, **overrides) -> Transformer:
    """Load a checkpoint directory produced by ``lite/convert.py``."""
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    args = ModelArgs.from_json(os.path.join(ckpt_path, "config.json"), **overrides)
    sd = load_state_dict_dir(ckpt_path)
    model = build_model(args, sd, device, dtype)
    del sd
    return model
