"""Data models shared by every stage of the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# "REVIEW" is never produced by the model. It is set by the code whenever the model
# output could not be used, so that a human always sees the gap instead of a made-up vote.
Vote = Literal["+", "-", "SPLIT", "ABSTAIN", "REVIEW"]


class Amendment(BaseModel):
    """One amendment as extracted from the committee PDF (no AI involved)."""

    number: int
    authors: str = ""
    doc_type: str = ""  # e.g. "Motion for a resolution"
    location: str = ""  # e.g. "Paragraph 3" or "Citation 6 a (new)"
    # Text keeps **...** around words printed in bold italic (the EP convention for changes).
    original_text: str = ""
    amendment_text: str = ""
    justification: str = ""
    language: str = ""  # original language code from the "Or. xx" line
    page: int = 0
    is_new: bool = False  # True when the left column is empty (pure insertion)
    relevance_score: float | None = None
    matched_priority: int | None = None  # 1-based index of the best matching priority

    @property
    def label(self) -> str:
        return f"AM {self.number}"


class ParseResult(BaseModel):
    amendments: list[Amendment]
    warnings: list[str] = Field(default_factory=list)
    expected_range: tuple[int, int] | None = None


class VoteDecision(BaseModel):
    """What the language model is asked to return for one amendment."""

    policy_rationale: str
    recommended_vote: Literal["+", "-", "SPLIT", "ABSTAIN"]


class Evaluation(BaseModel):
    """Final decision stored in the table (may be REVIEW when the model failed)."""

    recommended_vote: Vote
    policy_rationale: str
