"""Stage 2: rank amendments by semantic closeness to the NGO priorities.

This is a cheap filter that runs on CPU. Its only job is to decide which amendments are
worth the (expensive, GPU) language-model analysis. It does not decide how to vote.
"""

from __future__ import annotations

import re
from typing import Protocol
from collections.abc import Sequence

import numpy as np

from .parser import strip_markers
from .schemas import Amendment

MAX_EMBED_CHARS = 1500


class Embedder(Protocol):
    """Anything with a sentence-transformers style encode(); lets tests inject a stub."""

    def encode(self, sentences: Sequence[str], **kwargs): ...


def split_priorities(text: str) -> list[str]:
    """Split the priorities box into one item per line, dropping bullets and numbering.

    Scoring against each priority separately (and keeping the best match) works much
    better than embedding the whole box as one blob, because one long paragraph about
    three topics sits "in the middle" and matches none of them well.
    """
    items = []
    for raw in text.splitlines():
        item = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", raw).strip()
        if item:
            items.append(item)
    return items


def _embed(embedder: Embedder, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(embedder.encode(texts, batch_size=64), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def amendment_search_text(a: Amendment) -> str:
    """What gets embedded: where the amendment applies plus the proposed wording."""
    body = strip_markers(a.amendment_text) or strip_markers(a.original_text)
    return f"{a.location}. {body}"[:MAX_EMBED_CHARS]


def score_amendments(
    amendments: list[Amendment], priorities: str, embedder: Embedder
) -> list[Amendment]:
    """Set relevance_score (best cosine over all priorities) and matched_priority."""
    items = split_priorities(priorities)
    if not amendments or not items:
        return amendments
    prio_vecs = _embed(embedder, items)
    doc_vecs = _embed(embedder, [amendment_search_text(a) for a in amendments])
    sims = doc_vecs @ prio_vecs.T  # shape: (amendments, priorities)
    for a, row in zip(amendments, sims, strict=True):
        best = int(np.argmax(row))
        a.relevance_score = round(float(row[best]), 3)
        a.matched_priority = best + 1
    return amendments


def select_relevant(
    amendments: list[Amendment], threshold: float, max_items: int
) -> list[Amendment]:
    """Keep amendments at or above the threshold, best first, capped at max_items."""
    kept = [a for a in amendments if (a.relevance_score or 0.0) >= threshold]
    kept.sort(key=lambda a: a.relevance_score or 0.0, reverse=True)
    return kept[:max_items]
