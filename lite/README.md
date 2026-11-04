# DeepSeek-Lite: offline, single-machine inference

A compact rewrite of `inference/` for running DeepSeek MLA/MoE models on one
machine (CPU, NVIDIA GPU or Apple Silicon) with no server, no cluster and no
network access at run time.

## First, what fits on a local machine

The code can be made lean; the weights cannot be shrunk below what the
parameter count requires. From `python lite/estimate.py inference/configs/*.json`:

| Config | Total / active params | bf16 | int8 | int4 | KV cache, 4K ctx |
|---|---|---|---|---|---|
| 16B (DeepSeek-V2-Lite architecture) | 15.7 B / 2.4 B | 29.3 GiB | 14.6 GiB | **8.0 GiB** | 122 MiB |
| 236B (DeepSeek-V2) | 235.7 B / 20.9 B | 439 GiB | 220 GiB | 117 GiB | 270 MiB |
| 671B (DeepSeek-V3) | 671.0 B / 36.6 B | 1250 GiB | 625 GiB | 333 GiB | 274 MiB |

* **16B int4 (~8 GiB) or int8 (~15 GiB)** is the realistic laptop/desktop target.
  The matching public checkpoint is `deepseek-ai/DeepSeek-V2-Lite-Chat`.
* **DeepSeek-V3 (671B)** needs ~333 GiB even at int4. This code will run it on a
  single high-memory server, but not on a typical PC.
* Only the "active" parameters are used per token, so speed scales with the
  active count while memory scales with the total.

## Usage

```bash
pip install -r lite/requirements.txt

# one-time, offline after the download: convert + quantize
python lite/convert.py --hf-ckpt-path ./DeepSeek-V2-Lite-Chat --save-path ./v2-lite-int4 --quant int4

# chat (no network access is made)
python lite/chat.py --ckpt-path ./v2-lite-int4 --max-seq-len 4096 --stats
python lite/chat.py --ckpt-path ./v2-lite-int4 --input-file prompts.txt
```

`convert.py` also accepts the official FP8 DeepSeek-V3 checkpoint and dequantizes
it on the CPU (no GPU or Triton needed). `--quant none|int8|int4`; the output
head and embedding default to int8 when quantizing.

## What changed compared with `inference/`

| Issue in the reference demo | DeepSeek-Lite |
|---|---|
| Requires Triton + CUDA + NCCL/`torchrun`, Linux only | Pure PyTorch on CPU / CUDA / MPS |
| KV cache fixed at 8 x 16,384 tokens (3.8 GiB for 16B, 8.6 GiB for 671B) | Sized on request; default 1 x 4,096 (122-274 MiB) |
| bf16 or fp8 weights only | dense, int8 (per channel), int4 (group-wise, packed), block-FP8 |
| YaRN scaling switched on only when `max_seq_len > 4096`, so a smaller context changes the outputs | YaRN follows the model config |
| Causal mask wrong for multi-token forwards at `start_pos > 0` | Fixed; enables chunked prefill and KV reuse |
| Re-processes the whole conversation every turn | Reuses the cached prefix |
| Prompt longer than the context aborts with an assertion | Oldest turns are dropped |
| Router bias enabled by `dim == 7168` | Config flag (`gate_bias`), set by the converter |
| `fp8_cast_bf16.py` needs a CUDA GPU | Pure-torch FP8 dequant |
| Model built in RAM, then weights copied in (2x peak) | Built on the meta device; tensors assigned (~1x) |

Kept the same: the MLA weight-absorbed attention (caches only a 576-d latent
per token), the MoE routing, and the numerics. Unquantized, the outputs match
`inference/model.py` to 1e-4 in float32 (see tests).

## Learning from mistakes (`lite/memory.py`)

In chat:

* `/wrong <correct answer>`: saves (question, bad answer, correction) to
  `~/.deepseek_lite/feedback.jsonl` (change it with `--memory-file` or `DEEPSEEK_LITE_HOME`).
* On later questions, up to 3 corrections whose questions share enough
  keywords are put in front of your message as notes for the model.
* `/lessons` lists them, `/forget N` deletes one, and `/export FILE` writes them
  as a chat-format fine-tuning dataset (`{"messages": [...], "rejected": ...}`).
* Runtime errors during a turn are written to `errors.log` next to the memory
  file, and the session keeps going instead of crashing.

**Limits:** this is retrieval, not training.
* The model's weights never change, and it may still ignore a note.
* Matching is keyword-based, so a reworded question with no shared words will
  not find its correction.
* Turning the exported data into weight updates needs a separate fine-tuning
  tool and much more compute than inference.

## Tested vs. not tested

Tested (`python -m pytest lite/tests -q`, 16 tests, CPU):
* Equivalence with `inference/model.py` on small random V2-Lite-style and
  V3-style models (prefill and token-by-token decode).
* Chunked prefill equals full prefill; the reference's mask bug is demonstrated.
* int8, int4 and FP8 code paths are exact: the reference model given the same
  dequantized weights gives the same logits.
* End-to-end HF checkpoint → convert → load.
* Exact KV-cache reuse across turns, context trimming, and the memory store.
* Separately, `chat.py` was run end to end with an int4 tiny model and the
  `tokenizers`/`jinja2` fallback tokenizer.

Not tested (no GPU or real weights were available while writing this):
* Real checkpoints: output quality and how much int4 loses versus int8.
* Throughput on real hardware.
* The CUDA and MPS devices. MPS defaults to fp16, which may overflow for some
  inputs; use `--dtype fp32` if outputs look broken.

On tiny random models, logit cosine similarity against unquantized weights was
0.9986-0.9997 for int8 and 0.87-0.95 for int4. Random weights are a harsh case
and these numbers do not predict quality on trained weights. Measure on your
own tasks before relying on int4.

Quantization here is plain round-to-nearest. Calibrated methods (GPTQ/AWQ)
usually lose less at 4 bits but are not implemented. For maximum speed on CPU,
mature runtimes such as llama.cpp (GGUF) are likely faster than PyTorch eager
mode.
