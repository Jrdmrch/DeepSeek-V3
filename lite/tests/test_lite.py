"""
Tests for DeepSeek-Lite. Run from the repo root:  python -m pytest lite/tests -q

The equivalence tests compare lite/model.py against the original
inference/model.py on small random models in float32 on CPU. The original's
Triton kernels are never reached for bf16/fp32 weights, so a stub `kernel`
module is enough to import it.
"""
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "lite"))

import convert as lconvert  # noqa: E402
import estimate  # noqa: E402
from chat import Session, fit_history  # noqa: E402
from memory import FeedbackMemory, strip_notes  # noqa: E402
from model import (ModelArgs, QuantLinear, build_model, fp8_block_dequant, load_model,  # noqa: E402
                   quantize_int4, quantize_int8, unpack_int4)


def load_reference():
    sys.modules.setdefault("kernel", types.SimpleNamespace(act_quant=None, weight_dequant=None, fp8_gemm=None))
    spec = importlib.util.spec_from_file_location("ref_model", os.path.join(ROOT, "inference", "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = load_reference()

V2_LITE_TINY = dict(vocab_size=256, dim=64, inter_dim=96, moe_inter_dim=32, n_layers=3, n_dense_layers=1,
                    n_heads=4, n_routed_experts=8, n_shared_experts=2, n_activated_experts=2, q_lora_rank=0,
                    kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, mscale=0.707,
                    original_seq_len=16, rope_factor=4.0)
# dim=7168 is what makes the reference enable the router bias, so the V3-style test uses it
V3_TINY = dict(vocab_size=128, dim=7168, inter_dim=64, moe_inter_dim=32, n_layers=2, n_dense_layers=1,
               n_heads=2, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, n_expert_groups=4,
               n_limited_groups=2, score_func="sigmoid", route_scale=2.5, q_lora_rank=64, kv_lora_rank=32,
               qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16, original_seq_len=16, rope_factor=4.0)
SEQ = 48  # > original_seq_len, so the reference applies YaRN


def make_reference(cfg, seed=0):
    torch.manual_seed(seed)
    ref = REF.Transformer(REF.ModelArgs(max_batch_size=2, max_seq_len=SEQ, **cfg)).float()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif name.endswith("gate.bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
    return ref


def ref_state_dict(ref):
    return {k: v.detach().clone() for k, v in ref.state_dict().items()}


def lite_from(cfg, sd, max_seq_len=SEQ, batch=2):
    args = ModelArgs.from_dict(dict(cfg, max_seq_len=max_seq_len, max_batch_size=batch))
    return build_model(args, dict(sd), "cpu", torch.float32)


@pytest.mark.parametrize("cfg", [V2_LITE_TINY, V3_TINY], ids=["v2-lite-style", "v3-style"])
def test_matches_reference_prefill_and_decode(cfg):
    ref = make_reference(cfg)
    lite = lite_from(cfg, ref_state_dict(ref))
    torch.manual_seed(1)
    toks = torch.randint(0, cfg["vocab_size"], (2, 20))
    # prefill
    a, b = ref(toks[:, :12], 0), lite(toks[:, :12], 0)
    torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
    # token-by-token decode on top of the cache
    for pos in range(12, 20):
        a, b = ref(toks[:, pos:pos + 1], pos), lite(toks[:, pos:pos + 1], pos)
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)


def test_smaller_context_does_not_change_math():
    """Reference ties YaRN to max_seq_len; lite ties it to the config. Cache size 24 must equal 48."""
    ref = make_reference(V2_LITE_TINY)
    sd = ref_state_dict(ref)
    toks = torch.randint(0, 256, (1, 16))
    torch.testing.assert_close(lite_from(V2_LITE_TINY, sd, 24, 1)(toks), lite_from(V2_LITE_TINY, sd, 48, 1)(toks))


def test_chunked_prefill_matches_full_prefill():
    ref = make_reference(V3_TINY)
    lite = lite_from(V3_TINY, ref_state_dict(ref))
    toks = torch.randint(0, 128, (1, 30))
    full = lite(toks, 0)
    lite(toks[:, :11], 0)
    chunked = lite(toks[:, 11:], 11)
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


def test_reference_mask_bug_for_chunked_prefill():
    """Documents the bug fixed in lite: the reference cannot do a multi-token forward at start_pos > 0."""
    ref = make_reference(V2_LITE_TINY)
    toks = torch.randint(0, 256, (1, 20))
    full = ref(toks, 0)
    ref(toks[:, :8], 0)
    try:
        chunked = ref(toks[:, 8:], 8)
    except RuntimeError:
        return  # shape mismatch between the (12,12) mask and (12,20) scores
    assert not torch.allclose(chunked, full, atol=1e-4)


# ---------------------------------------------------------------- quantization

def test_int8_roundtrip():
    w = torch.randn(64, 256)
    q, s = quantize_int8(w)
    deq = q.float() * s[:, None]
    assert (deq - w).abs().max() <= s.max() / 2 + 1e-6
    lin = QuantLinear(256, 64)
    lin.set_weights(q, s, torch.float32, "cpu")
    x = torch.randn(3, 256)
    torch.testing.assert_close(lin(x), x @ deq.T, rtol=1e-5, atol=1e-5)


def test_int4_roundtrip_and_packing():
    w = torch.randn(32, 256)
    q, s = quantize_int4(w, 64)
    assert q.dtype == torch.uint8 and q.shape == (32, 128) and s.shape == (32, 4)
    deq = unpack_int4(q, s, torch.float32)
    err = (deq - w).view(32, 4, 64).abs().amax(-1)
    assert torch.all(err <= s / 2 + 1e-6)  # round-to-nearest bound per group
    lin = QuantLinear(256, 32)
    lin.set_weights(q, s, torch.float32, "cpu")
    x = torch.randn(2, 256)
    torch.testing.assert_close(lin(x), x @ deq.T, rtol=1e-5, atol=1e-5)


def test_fp8_block_dequant_matches_blockwise_definition():
    torch.manual_seed(0)
    M, N = 200, 300  # deliberately not multiples of 128
    w8 = (torch.randn(M, N) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand((M + 127) // 128, (N + 127) // 128) + 0.5
    expected = torch.empty(M, N)
    for i in range(s.size(0)):
        for j in range(s.size(1)):
            blk = (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
            expected[blk] = w8[blk].float() * s[i, j]
    torch.testing.assert_close(fp8_block_dequant(w8, s), expected)


def test_estimate_matches_instantiated_model():
    for cfg in (V2_LITE_TINY, V3_TINY):
        lite = lite_from(cfg, ref_state_dict(make_reference(cfg)))
        n = sum(t.numel() for t in list(lite.parameters()) + [b for name, b in lite.named_buffers()
                                                              if b is not None and "cache" not in name
                                                              and "rope" not in name])
        assert estimate.count_params(cfg)["total"] == n


# ---------------------------------------------------------------- converter

HF_RENAME = {v: k for k, v in lconvert.RENAME.items()}


def to_hf(sd, n_layers, fp8_names=()):
    """Turn a reference state dict into a Hugging Face style checkpoint (+ fake MTP layer)."""
    out = {}
    for name, t in sd.items():
        parts = name.split(".")
        if parts[-1] == "bias" and parts[-2] == "gate":
            parts[-1] = "e_score_correction_bias"
        else:
            parts[-2] = HF_RENAME[parts[-2]]
        hf = ".".join(parts).replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
        hf = hf if hf.startswith("lm_head") else "model." + hf
        if name in fp8_names:
            M, N = t.shape
            s = torch.rand((M + 127) // 128, (N + 127) // 128) * 0.01 + 0.005
            q = (t / s.repeat_interleave(128, 0)[:M].repeat_interleave(128, 1)[:, :N]).to(torch.float8_e4m3fn)
            out[hf] = q
            out[hf + "_scale_inv"] = s
            sd[name] = fp8_block_dequant(q, s)  # the reference sees the exact dequantized weight
        else:
            out[hf] = t.to(torch.bfloat16) if t.dim() == 2 else t
            if t.dim() == 2:
                sd[name] = out[hf].float()
    out[f"model.layers.{n_layers}.mlp.gate.weight"] = torch.zeros(1)  # MTP layer: must be skipped
    return out


def hf_config(cfg):
    return {"vocab_size": cfg["vocab_size"], "hidden_size": cfg["dim"], "intermediate_size": cfg["inter_dim"],
            "moe_intermediate_size": cfg["moe_inter_dim"], "num_hidden_layers": cfg["n_layers"],
            "first_k_dense_replace": cfg["n_dense_layers"], "num_attention_heads": cfg["n_heads"],
            "n_routed_experts": cfg["n_routed_experts"], "n_shared_experts": cfg["n_shared_experts"],
            "num_experts_per_tok": cfg["n_activated_experts"], "n_group": cfg.get("n_expert_groups", 1),
            "topk_group": cfg.get("n_limited_groups", 1), "scoring_func": cfg.get("score_func", "softmax"),
            "routed_scaling_factor": cfg.get("route_scale", 1.0),
            "topk_method": "noaux_tc" if cfg.get("score_func") == "sigmoid" else "greedy",
            "q_lora_rank": cfg["q_lora_rank"] or None, "kv_lora_rank": cfg["kv_lora_rank"],
            "qk_nope_head_dim": cfg["qk_nope_head_dim"], "qk_rope_head_dim": cfg["qk_rope_head_dim"],
            "v_head_dim": cfg["v_head_dim"], "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "rope_scaling": {"type": "yarn", "factor": cfg["rope_factor"], "beta_fast": 32, "beta_slow": 1,
                             "mscale": cfg.get("mscale", 1.0), "mscale_all_dim": cfg.get("mscale", 1.0),
                             "original_max_position_embeddings": cfg["original_seq_len"]}}


@pytest.mark.parametrize("cfg", [V2_LITE_TINY, V3_TINY], ids=["v2-lite-style", "v3-style"])
def test_convert_end_to_end(tmp_path, cfg):
    from safetensors.torch import save_file
    ref = make_reference(cfg)
    sd = ref_state_dict(ref)
    fp8 = [n for n in sd if sd[n].dim() == 2 and ".experts." in n] if cfg is V3_TINY else []
    hf = to_hf(sd, cfg["n_layers"], fp8)
    ref.load_state_dict(sd)
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    items = sorted(hf.items())
    save_file(dict(items[: len(items) // 2]), str(hf_dir / "a.safetensors"))
    save_file(dict(items[len(items) // 2:]), str(hf_dir / "b.safetensors"))
    (hf_dir / "config.json").write_text(json.dumps(hf_config(cfg)))
    (hf_dir / "tokenizer.json").write_text("{}")

    toks = torch.randint(0, cfg["vocab_size"], (1, 24))
    expected = ref(toks, 0)
    results = {}
    for quant in ("none", "int8", "int4"):
        out = tmp_path / quant
        lconvert.main(str(hf_dir), str(out), quant=quant, group_size=32, dtype="fp32", shard_gb=0.001)
        assert (out / "tokenizer.json").exists()
        model = load_model(str(out), "cpu", torch.float32, max_seq_len=SEQ)
        results[quant] = model(toks, 0)
    torch.testing.assert_close(results["none"], expected, rtol=1e-4, atol=1e-4)
    for quant, min_cos in (("int8", 0.995), ("int4", 0.9)):
        cos = torch.nn.functional.cosine_similarity(results[quant], expected).item()
        assert cos > min_cos, f"{quant}: cosine {cos:.4f}"


def test_hf_config_mapping_for_real_configs():
    v2_lite = {"vocab_size": 102400, "hidden_size": 2048, "intermediate_size": 10944, "moe_intermediate_size": 1408,
               "num_hidden_layers": 27, "first_k_dense_replace": 1, "num_attention_heads": 16,
               "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6, "n_group": 1,
               "topk_group": 1, "topk_method": "greedy", "scoring_func": "softmax", "routed_scaling_factor": 1.0,
               "q_lora_rank": None, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
               "v_head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000, "moe_layer_freq": 1,
               "rope_scaling": {"type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1, "mscale": 0.707,
                                "mscale_all_dim": 0.707, "original_max_position_embeddings": 4096}}
    with open(os.path.join(ROOT, "inference", "configs", "config_16B.json")) as f:
        demo = json.load(f)
    got = lconvert.hf_config_to_args(v2_lite)
    for k, v in demo.items():
        assert got[k] == pytest.approx(v), k
    assert got["gate_bias"] is False


# ---------------------------------------------------------------- generation

def greedy_no_cache_reuse(model, prompt, n):
    toks = list(prompt)
    for _ in range(n):
        toks.append(int(model(torch.tensor([toks]), 0).argmax()))
    return toks[len(prompt):]


def test_session_cache_reuse_is_exact():
    ref = make_reference(V2_LITE_TINY)
    model = lite_from(V2_LITE_TINY, ref_state_dict(ref), batch=1)
    s = Session(model, "cpu")
    p1 = [1, 2, 3, 4, 5]
    out1 = list(s.stream(p1, 6, None, 0.0, 1.0, prefill_chunk=2))
    assert out1 == greedy_no_cache_reuse(model, p1, 6)
    p2 = p1 + out1 + [7, 8, 9]  # next turn extends the previous one
    out2 = list(s.stream(p2, 5, None, 0.0, 1.0))
    assert out2 == greedy_no_cache_reuse(model, p2, 5)
    p3 = [9, 9] + p1  # a different prefix must invalidate the cache
    assert list(s.stream(p3, 4, None, 0.0, 1.0)) == greedy_no_cache_reuse(model, p3, 4)


def test_session_stops_at_eos_and_context_end():
    model = lite_from(V2_LITE_TINY, ref_state_dict(make_reference(V2_LITE_TINY)), max_seq_len=10, batch=1)
    s = Session(model, "cpu")
    assert len(list(s.stream([1, 2, 3, 4, 5, 6, 7, 8], 50, None, 0.0, 1.0))) == 3  # 10 - 8 + final token
    first = greedy_no_cache_reuse(model, [1, 2], 1)[0]
    s.reset()
    assert list(s.stream([1, 2], 5, first, 0.0, 1.0)) == []


class FakeTok:
    def chat_ids(self, messages, add_generation_prompt=True):
        return [0] * sum(len(m["content"]) for m in messages)


def test_fit_history_drops_oldest_exchange():
    msgs = [{"role": "user", "content": "a" * 10}, {"role": "assistant", "content": "b" * 10},
            {"role": "user", "content": "c" * 5}]
    assert len(fit_history(FakeTok(), msgs, 12)) == 5 and msgs[0]["content"] == "c" * 5
    with pytest.raises(ValueError):
        fit_history(FakeTok(), [{"role": "user", "content": "x" * 50}], 12)


# ---------------------------------------------------------------- memory

def test_memory_learns_and_persists(tmp_path):
    path = str(tmp_path / "fb.jsonl")
    m = FeedbackMemory(path)
    m.add("What is the capital of Australia?", "Sydney", "Canberra is the capital of Australia.")
    m.add("Write a python function to reverse a list", "def r(l): l.sort()", "Use l[::-1] or reversed(l).")
    hits = m.search("what's the capital city of australia")
    assert [h.correction for h in hits] == ["Canberra is the capital of Australia."]
    assert m.search("how do I bake sourdough bread") == []
    aug = m.augment("capital of Australia?")
    assert "Canberra" in aug and aug.endswith("capital of Australia?")
    assert strip_notes(aug) == "capital of Australia?"
    assert m.augment("unrelated question about cats") == "unrelated question about cats"

    with open(path, "a") as f:
        f.write("{not json\n")
    m2 = FeedbackMemory(path)  # reload tolerates a corrupt line
    assert len(m2.lessons) == 2 and m2.lessons[0].uses == 1
    m2.forget(0)
    assert len(FeedbackMemory(path).lessons) == 1
    n = m2.export_sft(str(tmp_path / "sft.jsonl"))
    rec = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert n == 1 and rec["messages"][1]["content"] == "Use l[::-1] or reversed(l)." and rec["rejected"]
    with pytest.raises(ValueError):
        m2.add("q", "a", "   ")
