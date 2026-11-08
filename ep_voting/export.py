"""Table building and Excel export."""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from .parser import strip_markers
from .schemas import Amendment, Evaluation

COLUMNS = [
    "Amendment",
    "Authors",
    "Location",
    "Vote",
    "Rationale",
    "Current text",
    "Amendment text",
    "Relevance",
    "Priority #",
]
_WIDTHS = [12, 32, 22, 8, 55, 60, 60, 10, 10]


def build_table(amendments: list[Amendment], decisions: list[Evaluation]) -> pd.DataFrame:
    """One row per amendment, in voting order (ascending amendment number)."""
    rows = []
    for a, d in zip(amendments, decisions, strict=True):
        rows.append(
            {
                "Amendment": a.label,
                "Authors": a.authors,
                "Location": f"{a.doc_type}: {a.location}" if a.doc_type else a.location,
                "Vote": d.recommended_vote,
                "Rationale": d.policy_rationale,
                "Current text": "(new)" if a.is_new else strip_markers(a.original_text),
                "Amendment text": strip_markers(a.amendment_text),
                "Relevance": a.relevance_score,
                "Priority #": a.matched_priority,
                "_n": a.number,
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS + ["_n"])
    return df.sort_values("_n").drop(columns="_n").reset_index(drop=True)


def write_excel(df: pd.DataFrame, directory: str | None = None) -> str:
    """Write the (possibly user-edited) table to a fresh .xlsx and return its path."""
    from openpyxl.styles import Alignment, Font, PatternFill

    folder = directory or tempfile.mkdtemp(prefix="ep_voting_")
    path = os.path.join(folder, "EP_Voting_List.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Voting list")
        ws = writer.sheets["Voting list"]
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="334155")
        for idx, width in enumerate(_WIDTHS[: len(df.columns)], start=1):
            ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    return path
