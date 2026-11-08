"""Stage 3: ask a language model for a vote recommendation, and read its answer safely.

Design choices:
* The model only decides the vote and the rationale. Authors, location and texts were
  already extracted by the parser, so the model cannot corrupt them.
* The answer is forced to start as JSON by pre-filling the beginning of the reply.
* Any answer that cannot be read becomes the vote "REVIEW". The code never invents a
  vote to fill a gap.
* The model call itself is injected (a plain function), so this module needs no GPU
  and can be tested with a fake generator.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections.abc import Callable, Sequence

from .diff import describe_change
from .parser import strip_markers
from .schemas import Amendment, Evaluation, VoteDecision

# The assistant reply starts with this text, so the model just continues the JSON.
# The rationale comes FIRST and the vote LAST: the model has to work out what the
# amendment does before it commits to a vote. (Voting first made a small model pick
# SPLIT almost every time.)
PREFILL = '{"policy_rationale": "'

MAX_FIELD_CHARS = 1600
MAX_CHANGE_CHARS = 700
MAX_RATIONALE_WORDS = 45

SYSTEM_PROMPT = """You are a senior EU legislative adviser supporting the policy team of an NGO.
You receive the NGO's priorities and red lines and ONE amendment tabled in a European Parliament committee. Decide how the NGO should vote on that amendment.

How to work:
1. Find out exactly what the amendment changes. Words under REMOVED are deleted by the amendment, words under ADDED are inserted by it. Read the direction carefully.
2. Check whether that change strengthens or weakens a priority. A priority is affected only if the change really touches it. Do not stretch a priority to cover an unrelated topic.
3. Vote.

Votes (write the word, not a symbol):
SUPPORT the change strengthens or protects a priority and harms none
OPPOSE  the change weakens a priority, removes a protection, or does what a red line forbids
ABSTAIN the change does not clearly affect any priority, it is only a drafting or wording change, or the effect on a priority is only speculative (you would need words like "could", "might" or "potentially" to describe it)
SPLIT   ONLY if the amendment contains two clearly separate changes, one the NGO supports and one it opposes, and you can name both. A new recital, a one-sentence paragraph or a single change is never SPLIT.

The vote must follow from your rationale. If your rationale says the amendment removes a protection or a funding commitment, the vote is OPPOSE. If it says the amendment adds one, the vote is SUPPORT.

Examples (for an NGO that wants dedicated youth funding and direct grant access for NGOs):
- REMOVED: "dedicated funding for apprenticeships" -> OPPOSE (it takes away dedicated funding).
- ADDED: "and civil society organisations" to the list of eligible beneficiaries -> SUPPORT (it widens NGO access).
- REMOVED: "rapidly" ADDED: "swiftly" -> ABSTAIN (wording only).

Answer with one JSON object with exactly these two keys, in this order: "policy_rationale" (at most 35 words: what changes and which priority it affects) and then "recommended_vote", which must be exactly one of SUPPORT, OPPOSE, ABSTAIN, SPLIT. Output nothing else."""

Generator = Callable[[list[list[dict]]], list[str]]

_VOTE_ALIASES = {
    "+": "+", "plus": "+", "for": "+", "yes": "+", "in favour": "+", "in favor": "+",
    "support": "+", "approve": "+",
    "-": "-", "against": "-", "no": "-", "oppose": "-", "reject": "-",
    "split": "SPLIT", "split vote": "SPLIT",
    "abstain": "ABSTAIN", "abstention": "ABSTAIN", "0": "ABSTAIN", "neutral": "ABSTAIN",
}


def _clip_to(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


def _clip(text: str) -> str:
    return _clip_to(text, MAX_FIELD_CHARS)


def build_messages(a: Amendment, priorities: str) -> list[dict]:
    """Chat messages for one amendment (the tokenizer's chat template is applied later)."""
    removed, added = describe_change(a)
    parts = [
        "NGO PRIORITIES AND RED LINES:",
        priorities.strip(),
        "",
        f"AMENDMENT {a.number} ({a.doc_type}, {a.location})",
        f"Authors: {a.authors}",
    ]
    if a.is_new:
        parts += ["This is a NEW INSERTION. Text inserted:", _clip(added)]
    else:
        parts += [
            "REMOVED by the amendment: " + (_clip_to(removed, MAX_CHANGE_CHARS) or "(nothing)"),
            "ADDED by the amendment: " + (_clip_to(added, MAX_CHANGE_CHARS) or "(nothing)"),
            "",
            "Full text after the amendment:",
            _clip(strip_markers(a.amendment_text)) if added or removed else "(unchanged)",
        ]
    if a.justification:
        parts += ["", "Justification:", _clip(a.justification)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _normalise_vote(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace("−", "-").replace("–", "-")
    return _VOTE_ALIASES.get(v)


def _limit_words(text: str) -> str:
    words = text.split()
    return text.strip() if len(words) <= MAX_RATIONALE_WORDS else " ".join(words[:MAX_RATIONALE_WORDS]) + "..."


def parse_model_output(completion: str) -> Evaluation | None:
    """Turn the raw model continuation into a decision, or None if unusable.

    Accepts the continuation of PREFILL, a full JSON object, or JSON wrapped in code
    fences. A reply without a readable vote is unusable: the code never guesses one.
    """
    # Two readings of the same text: a full JSON object (possibly after code fences), or
    # a continuation of the pre-filled opening.
    for text in (completion, PREFILL + completion):
        start = text.find("{")
        if start < 0:
            continue
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        vote = _normalise_vote(data.get("recommended_vote"))
        rationale = data.get("policy_rationale")
        if vote and isinstance(rationale, str) and rationale.strip():
            decision = VoteDecision(policy_rationale=_limit_words(rationale), recommended_vote=vote)
            return Evaluation(**decision.model_dump())
    # Cut off or malformed JSON: if the vote is still readable, keep it but flag the row.
    m = re.search(r'"recommended_vote"\s*:\s*"([^"]*)"', completion)
    vote = _normalise_vote(m.group(1)) if m else None
    if vote:
        before = completion[: m.start()].strip().rstrip(",").strip().rstrip('"')
        note = "[Check manually, answer was malformed] " + _limit_words(before)
        return Evaluation(recommended_vote=vote, policy_rationale=note.strip())
    return None


def _review(reason: str) -> Evaluation:
    return Evaluation(recommended_vote="REVIEW", policy_rationale=reason)


def _generate_with_retry(generate: Generator, prompts: list[list[dict]], retries: int, wait: float) -> list[str]:
    """Call the model, retrying a few times: ZeroGPU sometimes fails to attach a GPU and
    works a few seconds later. Quota errors are not retried, they will not fix themselves."""
    for attempt in range(retries + 1):
        try:
            return generate(prompts)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries or "quota" in str(exc).lower():
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")


def evaluate_amendments(
    amendments: Sequence[Amendment],
    priorities: str,
    generate: Generator,
    chunk_size: int = 24,
    on_progress: Callable[[int, int], None] | None = None,
    retries: int = 2,
    retry_wait: float = 4.0,
) -> tuple[list[Evaluation], list[str]]:
    """Evaluate amendments in chunks. Returns (decisions aligned with input, error messages).

    Each chunk is one call to `generate`, which on ZeroGPU is one GPU allocation. If a
    call fails (for example the GPU quota runs out), the failed chunk and everything after
    it is marked REVIEW so the user keeps all results obtained so far.
    """
    results: list[Evaluation] = []
    errors: list[str] = []
    total = len(amendments)
    for start in range(0, total, chunk_size):
        chunk = amendments[start : start + chunk_size]
        if on_progress:
            on_progress(start, total)
        try:
            prompts = [build_messages(a, priorities) for a in chunk]
            completions = _generate_with_retry(generate, prompts, retries, retry_wait)
            if len(completions) != len(chunk):
                raise RuntimeError("Model returned a different number of answers than prompts.")
        except Exception as exc:  # noqa: BLE001, GPU errors must not lose finished chunks
            traceback.print_exc()  # full details go to the Space logs
            detail = f"{type(exc).__name__}: {exc}"
            errors.append(f"Stopped at amendment {chunk[0].number}: {detail}")
            results.extend(_review(f"Not analysed ({detail})") for _ in range(total - start))
            return results, errors
        for completion in completions:
            parsed = parse_model_output(completion)
            if parsed is None:
                snippet = " ".join(completion.split())[:160]
                parsed = _review(f"The model answer could not be read. Review manually. Raw start: {snippet!r}")
            results.append(parsed)
    if on_progress:
        on_progress(total, total)
    return results, errors
