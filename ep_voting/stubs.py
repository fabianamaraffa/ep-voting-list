"""Tiny fake models used by tests and by `EP_USE_STUBS=1` (no downloads, no GPU)."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import numpy as np


class StubEmbedder:
    """Hashed bag-of-words vectors: similar wording gives a high cosine, no model needed."""

    dim = 256

    def encode(self, sentences: Sequence[str], **_: object) -> np.ndarray:
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            for word in re.findall(r"[a-z0-9+]+", s.lower()):
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        return out


def stub_generate(prompts: list[list[dict]]) -> list[str]:
    """Returns the continuation of the JSON prefill, using trivial keyword rules."""
    answers = []
    for messages in prompts:
        text = messages[-1]["content"].lower()
        if "doubling" in text and "increasing" in text:
            vote, why = "-", "Weakens the call to double ESF+ funding."
        elif "youth" in text or "apprentice" in text:
            vote, why = "+", "Supports the youth employment priority."
        else:
            vote, why = "ABSTAIN", "No clear link to the priorities."
        answers.append(f'{why}", "recommended_vote": "{vote}"}}')
    return answers
