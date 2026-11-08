"""The whole workflow in one function, independent of Gradio and of the real models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd

from .export import COLUMNS, build_table
from .llm import Generator, evaluate_amendments
from .parser import parse_pdf
from .screening import Embedder, score_amendments, select_relevant, split_priorities

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineOutput:
    table: pd.DataFrame
    status: str


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def run_pipeline(
    pdf_path: str | None,
    priorities: str,
    threshold: float,
    max_items: int,
    embedder: Embedder,
    generate: Generator,
    chunk_size: int = 24,
    progress: ProgressFn | None = None,
) -> PipelineOutput:
    """parse -> score -> select -> LLM evaluation -> table."""

    def tick(fraction: float, text: str) -> None:
        if progress:
            progress(fraction, text)

    if not pdf_path:
        return PipelineOutput(_empty(), "Please upload the amendments PDF.")
    if not split_priorities(priorities):
        return PipelineOutput(_empty(), "Please enter the NGO priorities (one per line).")

    tick(0.02, "Reading the PDF")
    parsed = parse_pdf(pdf_path)
    if not parsed.amendments:
        return PipelineOutput(_empty(), "No amendments found. " + " ".join(parsed.warnings))

    tick(0.15, "Scoring relevance")
    score_amendments(parsed.amendments, priorities, embedder)
    selected = select_relevant(parsed.amendments, threshold, max_items)
    above = sum(1 for a in parsed.amendments if (a.relevance_score or 0) >= threshold)
    if not selected:
        return PipelineOutput(
            _empty(),
            f"{len(parsed.amendments)} amendments read, none reached the relevance threshold "
            f"({threshold:.2f}). Lower the threshold and run again.",
        )

    def on_progress(done: int, total: int) -> None:
        tick(0.25 + 0.7 * done / max(total, 1), f"Analysing amendments ({done}/{total})")

    decisions, errors = evaluate_amendments(
        selected, priorities, generate, chunk_size=chunk_size, on_progress=on_progress
    )
    table = build_table(selected, decisions)

    lines = [f"**{len(selected)}** amendments analysed out of {len(parsed.amendments)} read."]
    if above > len(selected):
        lines.append(
            f"{above - len(selected)} more passed the threshold but were left out by the "
            f"limit of {max_items}. Raise the limit to include them."
        )
    n_review = int((table["Vote"] == "REVIEW").sum())
    if n_review:
        lines.append(f"**{n_review}** rows are marked REVIEW and need a manual decision.")
    lines += [f"Warning: {w}" for w in parsed.warnings]
    lines += [f"Error: {e}" for e in errors]
    lines.append("This is a draft to support your decision. Check every recommendation before use.")
    return PipelineOutput(table, "\n\n".join(lines))
