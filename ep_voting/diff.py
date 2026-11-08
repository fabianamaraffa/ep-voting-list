"""What exactly does an amendment change? Computed by code, not guessed by the model.

In an EP amendment table both columns mark changed words in bold italic: in the left
column those words are the ones being replaced, in the right column the ones being
introduced. A small model confuses the two directions, so we turn those marks into a
plain "removed / added" description and give that to the model.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .parser import strip_markers
from .schemas import Amendment

_DELETION_WORDS = {"deleted", "deleted.", "deleted;"}
_RUN_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _marked_runs(text: str) -> str:
    return " | ".join(" ".join(m.split()) for m in _RUN_RE.findall(text))


def describe_change(a: Amendment) -> tuple[str, str]:
    """Return (removed, added) as short strings. Changes are joined with ' | '.

    The EP prints changed words in bold italic: in the left column the words being
    replaced, in the right column the words being introduced. Those marks are the
    official statement of what changes, so they are used when present. A word-level diff
    is the fallback for amendments that do not follow the convention.
    """
    original = strip_markers(a.original_text).split()
    amended = strip_markers(a.amendment_text).split()
    if not original:
        return "", " ".join(amended)  # pure insertion
    if len(amended) == 1 and amended[0].lower() in _DELETION_WORDS:
        return " ".join(original), ""  # the whole text is deleted
    marked_removed = _marked_runs(a.original_text)
    marked_added = _marked_runs(a.amendment_text)
    if marked_removed or marked_added:
        return marked_removed, marked_added
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, original, amended, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed.append(" ".join(original[i1:i2]))
        if tag in ("replace", "insert"):
            added.append(" ".join(amended[j1:j2]))
    return " | ".join(removed), " | ".join(added)
