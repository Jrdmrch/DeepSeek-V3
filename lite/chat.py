"""
Offline chat / batch generation for DeepSeek-Lite checkpoints.

    python lite/chat.py --ckpt-path ./v2-lite-int4                 # interactive
    python lite/chat.py --ckpt-path ./v2-lite-int4 --input-file prompts.txt

Nothing here touches the network: Hugging Face libraries are forced into
offline mode and the tokenizer is read from the checkpoint directory.

Chat commands:
    /wrong [correction]  mark the last answer as wrong and save a correction
    /lessons             list saved corrections      /forget N   delete one
    /export FILE         write corrections as a fine-tuning dataset (JSONL)
    /clear               start a new conversation    /exit       quit
"""
import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from typing import Iterator, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import FeedbackMemory, default_memory_path, strip_notes  # noqa: E402
from model import Transformer, default_device, default_dtype, load_model  # noqa: E402


# --------------------------------------------------------------------------
# Tokenizer: transformers if installed, otherwise `tokenizers` + jinja2
# --------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, path: str):
        try:
            from transformers import AutoTokenizer
            self._hf = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.eos_id = self._hf.eos_token_id
            return
        except ImportError:
            self._hf = None
        from tokenizers import Tokenizer as RawTokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self._tok = RawTokenizer.from_file(os.path.join(path, "tokenizer.json"))
        with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)

        def tok_str(v):
            return v.get("content") if isinstance(v, dict) else v

        self.bos = tok_str(cfg.get("bos_token")) or ""
        self.eos = tok_str(cfg.get("eos_token")) or ""
        self.eos_id = self._tok.token_to_id(self.eos)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise ValueError(msg)

        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(cfg["chat_template"])

    def chat_ids(self, messages, add_generation_prompt=True) -> List[int]:
        if self._hf is not None:
            ids = self._hf.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
            return list(ids["input_ids"] if isinstance(ids, dict) else ids)
        text = self._template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                     bos_token=self.bos, eos_token=self.eos)
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        if self._hf is not None:
            return self._hf.decode(ids, skip_special_tokens=True)
        return self._tok.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = probs.sort(dim=-1, descending=True)
        drop = sorted_p.cumsum(dim=-1) - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_i, sorted_p)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1).squeeze(-1)


class Session:
    """Single-sequence generator that reuses the KV cache for a shared prompt prefix."""

    def __init__(self, model: Transformer, device):
        self.model = model
        self.device = device
        self.cached: List[int] = []  # tokens whose keys/values are currently in the cache

    def reset(self):
        self.cached = []

    def stream(self, prompt: List[int], max_new_tokens: int, eos_id: Optional[int],
               temperature: float, top_p: float, prefill_chunk: int = 512) -> Iterator[int]:
        n = 0
        while n < min(len(prompt), len(self.cached)) and prompt[n] == self.cached[n]:
            n += 1
        n = min(n, len(prompt) - 1)  # always run at least one token to get fresh logits
        self.cached = self.cached[:n]
        pos, logits = n, None
        while pos < len(prompt):  # chunked prefill bounds peak activation memory
            chunk = prompt[pos:pos + prefill_chunk]
            logits = self.model(torch.tensor([chunk], device=self.device), pos)
            self.cached += chunk
            pos += len(chunk)
        for i in range(max_new_tokens):
            tok = int(sample(logits[0], temperature, top_p))
            if tok == eos_id:
                return
            yield tok
            if i == max_new_tokens - 1 or pos >= self.model.max_seq_len:
                return
            logits = self.model(torch.tensor([[tok]], device=self.device), pos)
            self.cached.append(tok)
            pos += 1


def fit_history(tok: Tokenizer, messages: list, budget: int) -> List[int]:
    """Drop the oldest exchanges until the prompt fits in ``budget`` tokens."""
    while True:
        ids = tok.chat_ids(messages)
        if len(ids) <= budget:
            return ids
        if len(messages) <= 1:
            raise ValueError(f"Prompt is {len(ids)} tokens; the context budget is {budget}. "
                             f"Shorten it or raise --max-seq-len / lower --max-new-tokens.")
        del messages[: 2 if messages[0]["role"] == "user" and len(messages) > 2 else 1]


def respond(session, tok, messages, args, out=sys.stdout) -> str:
    budget = session.model.max_seq_len - args.max_new_tokens
    ids = fit_history(tok, messages, budget)
    gen: List[int] = []
    printed = ""
    t0 = time.time()
    for t in session.stream(ids, args.max_new_tokens, tok.eos_id, args.temperature, args.top_p):
        gen.append(t)
        text = tok.decode(gen)
        if not text.endswith("�"):  # wait for complete multi-byte characters
            out.write(text[len(printed):])
            out.flush()
            printed = text
    text = tok.decode(gen)
    out.write(text[len(printed):] + "\n")
    if args.stats:
        dt = time.time() - t0
        out.write(f"[{len(gen)} tokens in {dt:.1f}s, {len(gen) / max(dt, 1e-9):.2f} tok/s, prompt {len(ids)} tokens]\n")
    return text


def log_error(memory_dir: str, exc: BaseException):
    os.makedirs(memory_dir, exist_ok=True)
    with open(os.path.join(memory_dir, "errors.log"), "a", encoding="utf-8") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{''.join(traceback.format_exception(exc))}\n")


def interactive(session, tok, args, memory: Optional[FeedbackMemory]):
    messages: list = []
    last = None  # (original user prompt, answer)
    mem_dir = os.path.dirname(os.path.abspath(args.memory_file))
    print("Type a message. Commands: /wrong [fix], /lessons, /forget N, /export FILE, /clear, /exit")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "/exit":
            break
        if cmd == "/clear":
            messages.clear()
            session.reset()
            last = None
            continue
        if cmd in ("/wrong", "/lessons", "/forget", "/export") and memory is None:
            print("Memory is disabled (--no-memory).")
            continue
        if cmd == "/wrong":
            if last is None:
                print("Nothing to correct yet.")
                continue
            fix = rest.strip() or input("What should the answer have been? ").strip()
            if not fix:
                print("No correction given; nothing saved.")
                continue
            memory.add(last[0], last[1], fix)
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = fix  # the conversation continues from the corrected answer
            print(f"Saved. {len(memory.lessons)} correction(s) in {memory.path}")
            continue
        if cmd == "/lessons":
            for i, l in enumerate(memory.lessons):
                print(f"[{i}] Q: {l.prompt[:80]!r}\n    fix: {l.correction[:120]!r} (used {l.uses}x)")
            if not memory.lessons:
                print("No corrections saved.")
            continue
        if cmd == "/forget":
            try:
                l = memory.forget(int(rest))
                print(f"Deleted correction for: {l.prompt[:80]!r}")
            except (ValueError, IndexError):
                print("Usage: /forget N (see /lessons)")
            continue
        if cmd == "/export":
            if not rest.strip():
                print("Usage: /export FILE")
                continue
            print(f"Wrote {memory.export_sft(rest.strip())} records to {rest.strip()}")
            continue

        content = memory.augment(line) if memory else line
        messages.append({"role": "user", "content": content})
        try:
            answer = respond(session, tok, messages, args)
        except Exception as e:  # keep the session alive; record what went wrong
            messages.pop()
            session.reset()
            log_error(mem_dir, e)
            print(f"[error] {type(e).__name__}: {e}  (details in {os.path.join(mem_dir, 'errors.log')})")
            continue
        messages.append({"role": "assistant", "content": answer})
        last = (strip_notes(line), answer)


def main():
    p = ArgumentParser(description="Offline DeepSeek-Lite chat")
    p.add_argument("--ckpt-path", required=True, help="directory produced by lite/convert.py")
    p.add_argument("--input-file", default="", help="one prompt per line; omit for interactive chat")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: best available)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    p.add_argument("--max-seq-len", type=int, default=4096, help="KV cache length (context window)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--memory-file", default=default_memory_path(), help="where corrections are stored")
    p.add_argument("--no-memory", action="store_true", help="disable the correction memory")
    p.add_argument("--stats", action="store_true", help="print tokens/s after each answer")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or default_device()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(args.dtype) or default_dtype(device)
    t0 = time.time()
    model = load_model(args.ckpt_path, device, dtype, max_seq_len=args.max_seq_len, max_batch_size=1)
    print(f"Loaded in {time.time() - t0:.1f}s on {device} ({dtype}), context {args.max_seq_len} tokens")
    tok = Tokenizer(args.ckpt_path)
    session = Session(model, device)
    memory = None if args.no_memory else FeedbackMemory(args.memory_file)

    if not args.input_file:
        interactive(session, tok, args, memory)
        return
    with open(args.input_file, encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    for prompt in prompts:  # sequential: batch 1 keeps memory minimal
        print("Prompt:", prompt)
        print("Completion: ", end="")
        session.reset()
        content = memory.augment(prompt) if memory else prompt
        respond(session, tok, [{"role": "user", "content": content}], args)
        print()


if __name__ == "__main__":
    main()
