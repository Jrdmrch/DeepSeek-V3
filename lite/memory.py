"""
Local "learn from mistakes" memory for DeepSeek-Lite.

What this does
--------------
When you mark an answer as wrong and supply a correction, the (prompt, bad
answer, correction) triple is appended to a local JSONL file. On later turns,
corrections whose original prompts are similar to the new prompt are retrieved
(keyword/IDF overlap, no extra model or internet needed) and placed in front of
the user's message so the model can take them into account.

What this does NOT do
---------------------
It does not change the model's weights. The model "learns" only in the sense
that relevant past corrections are put back in its context window; it can still
ignore or misapply them, and retrieval is keyword-based, so a paraphrased
question with no shared words will not match. Corrections can be exported as a
supervised fine-tuning dataset (``export_sft``) if you later want to train the
weights with a separate tool.
"""
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import List, Optional

_WORD = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]", re.IGNORECASE)
_STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been
being it its this that these those i you he she we they me my your our their what which who whom how why
when where do does did can could should would will shall may might must not no yes so than too very just
about into over under again please tell give show explain""".split())


def _terms(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD.findall(text)]
    # keep words (and single CJK/other symbols, which carry meaning in non-space-separated scripts)
    return [t for t in toks if t not in _STOP and (len(t) > 1 or not t.isascii())]


@dataclass
class Lesson:
    prompt: str
    bad_response: str
    correction: str
    created: float = field(default_factory=time.time)
    uses: int = 0


class FeedbackMemory:
    def __init__(self, path: str, max_lessons_in_prompt: int = 3, min_score: float = 0.35, max_chars: int = 600):
        self.path = path
        self.max_lessons = max_lessons_in_prompt
        self.min_score = min_score
        self.max_chars = max_chars
        self.lessons: List[Lesson] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.lessons.append(Lesson(**json.loads(line)))
                    except (ValueError, TypeError) as e:  # keep going if one line is corrupt
                        print(f"[memory] skipping bad line {line_no} in {path}: {e}")

    # ---- persistence -----------------------------------------------------
    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for l in self.lessons:
                f.write(json.dumps(asdict(l), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)  # atomic: a crash cannot leave a half-written file

    def add(self, prompt: str, bad_response: str, correction: str) -> Lesson:
        if not correction.strip():
            raise ValueError("correction must not be empty")
        lesson = Lesson(prompt=prompt, bad_response=bad_response, correction=correction.strip())
        self.lessons.append(lesson)
        self._save()
        return lesson

    def forget(self, index: int) -> Lesson:
        lesson = self.lessons.pop(index)
        self._save()
        return lesson

    # ---- retrieval -------------------------------------------------------
    def search(self, query: str) -> List[Lesson]:
        """Lessons whose prompt shares enough IDF-weighted terms with ``query``."""
        q = set(_terms(query))
        if not q or not self.lessons:
            return []
        docs = [set(_terms(l.prompt)) for l in self.lessons]
        n = len(docs)
        df = Counter(t for d in docs for t in d)
        idf = {t: math.log(1 + (n + 1) / (df.get(t, 0) + 0.5)) for t in q}
        total = sum(idf.values())
        scored = []
        for i, d in enumerate(docs):
            overlap = q & d
            if not overlap:
                continue
            # fraction of the query's information covered, penalised for very different lengths
            score = sum(idf[t] for t in overlap) / total
            score *= min(1.0, 2 * len(overlap) / max(len(d), 1)) ** 0.5
            if score >= self.min_score:
                scored.append((score, -i, i))
        scored.sort(reverse=True)
        return [self.lessons[i] for _, _, i in scored[: self.max_lessons]]

    def augment(self, user_prompt: str) -> str:
        """Return the user prompt, prefixed with relevant past corrections if any."""
        hits = self.search(user_prompt)
        if not hits:
            return user_prompt
        for h in hits:
            h.uses += 1
        self._save()
        notes = []
        for h in hits:
            notes.append(f"- Earlier question: {h.prompt[:200]}\n  A previous answer was marked wrong. "
                         f"Correct answer/guidance: {h.correction[: self.max_chars]}")
        return ("[Notes from earlier corrections by this user. Apply them if relevant; ignore them if not.]\n"
                + "\n".join(notes) + "\n[End of notes]\n\n" + user_prompt)

    # ---- export ----------------------------------------------------------
    def export_sft(self, path: str) -> int:
        """Write corrections as chat-format SFT records: {"messages": [user, assistant]}."""
        with open(path, "w", encoding="utf-8") as f:
            for l in self.lessons:
                rec = {"messages": [{"role": "user", "content": l.prompt},
                                    {"role": "assistant", "content": l.correction}],
                       "rejected": l.bad_response}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(self.lessons)


def default_memory_path() -> str:
    base = os.environ.get("DEEPSEEK_LITE_HOME", os.path.join(os.path.expanduser("~"), ".deepseek_lite"))
    return os.path.join(base, "feedback.jsonl")


def strip_notes(text: str) -> Optional[str]:
    """Remove an injected notes block from a stored prompt (used when saving a new lesson)."""
    marker = "[End of notes]\n\n"
    return text.split(marker, 1)[1] if marker in text else text
