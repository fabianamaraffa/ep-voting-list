"""Deterministic parser for European Parliament committee amendment PDFs.

The EP publishes amendments as a PDF where every amendment looks like this:

    Amendment 87                              <- bold heading, alone on its line
    Miriam Lexmann, Jagna Marczulajtis-Walczak  <- authors (may wrap on several lines)
    Motion for a resolution                   <- document type (bold)
    Paragraph 3                               <- location (bold)
        Motion for a resolution | Amendment   <- italic column labels
        <current text>          | <amended text, changes in bold italic>
    Or. en                                    <- end marker, original language

Amendments often run across a page break, and the heading can even be separated from
its author line by one. A page-by-page regex on flattened text breaks in those cases,
so this parser reads text lines with their coordinates and runs a small state machine
that keeps its state across pages. Columns are separated by x position, which also
means the two columns never get interleaved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from .schemas import Amendment, ParseResult

AMENDMENT_RE = re.compile(r"^Amendment\s+(\d+)$")
END_RE = re.compile(r"^Or\.\s*([A-Za-z]{2})$")
COVER_RANGE_RE = re.compile(r"AMENDMENTS\s+(\d+)\s*[-–]\s*(\d+)")
# Running header/footer lines: "PE766.605v02-00", "42/158", "AM\1310525EN.docx"
FOOTER_RE = re.compile(r"^(PE\d+\.\d+v\d+-\d+|\d+/\d+|AM\\\S+)$")

FOOTER_Y = 0.88  # lines starting below 88% of the page height are footer
HEADER_Y = 0.04  # lines ending above 4% of the page height are header
MARK_RE = re.compile(r"\*\*")

IDLE, HEADER, TABLE, JUSTIFICATION = "idle", "header", "table", "justification"


@dataclass
class _Line:
    plain: str
    marked: str  # same text with **...** around bold-italic runs
    x0: float
    y0: float
    y1: float
    page: int
    bold: bool
    italic: bool
    all_marked: bool


@dataclass
class _Builder:
    number: int
    page: int
    header: list[_Line] = field(default_factory=list)
    left: list[_Line] = field(default_factory=list)
    right: list[_Line] = field(default_factory=list)
    just: list[_Line] = field(default_factory=list)
    left_label: str = ""
    language: str = ""


def strip_markers(text: str) -> str:
    """Remove the ** change markers (used for the table and the Excel export)."""
    return MARK_RE.sub("", text)


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _is_italic(span: dict) -> bool:
    font = span["font"].lower()
    return bool(span["flags"] & 2) or "italic" in font or "oblique" in font


def _build_line(raw: dict, page_no: int) -> _Line | None:
    spans = raw["spans"]
    plain = "".join(s["text"] for s in spans).strip()
    if not plain:
        return None
    # Mark each non-blank span; a blank span between two marked spans stays inside the run.
    flags: list[bool | None] = []
    for s in spans:
        flags.append(None if not s["text"].strip() else (_is_bold(s) and _is_italic(s)))
    for i, f in enumerate(flags):
        if f is None:
            prev = next((flags[j] for j in range(i - 1, -1, -1) if flags[j] is not None), False)
            nxt = next((flags[j] for j in range(i + 1, len(flags)) if flags[j] is not None), False)
            flags[i] = bool(prev and nxt)
    marked_parts: list[str] = []
    in_run = False
    for s, f in zip(spans, flags, strict=True):
        if f and not in_run:
            marked_parts.append("**")
            in_run = True
        elif not f and in_run:
            marked_parts.append("**")
            in_run = False
        marked_parts.append(s["text"])
    if in_run:
        marked_parts.append("**")
    real = [s for s in spans if s["text"].strip()]
    x0, y0, _, y1 = raw["bbox"]
    return _Line(
        plain=plain,
        marked="".join(marked_parts).strip(),
        x0=x0,
        y0=y0,
        y1=y1,
        page=page_no,
        bold=all(_is_bold(s) for s in real),
        italic=all(_is_italic(s) for s in real),
        all_marked=all(_is_bold(s) and _is_italic(s) for s in real),
    )


def _page_lines(page: pymupdf.Page, page_no: int) -> list[_Line]:
    height = page.rect.height
    lines: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for raw in block["lines"]:
            line = _build_line(raw, page_no)
            if line is None:
                continue
            if line.y0 > FOOTER_Y * height or line.y1 < HEADER_Y * height:
                continue
            if FOOTER_RE.match(line.plain):
                continue
            lines.append(line)
    # Reading order: top to bottom, then left to right. Bucketing y into 3pt rows makes
    # the two column labels of the same row sort by x instead of by float noise.
    lines.sort(key=lambda ln: (round(ln.y0 / 3), ln.x0))
    return lines


def _join(lines: list[_Line], use_markers: bool = True) -> str:
    """Join lines into text: de-hyphenate wraps, keep paragraph breaks, merge ** runs."""
    if not lines:
        return ""
    if use_markers and all(ln.all_marked for ln in lines):
        use_markers = False  # a column that is entirely new text needs no markers
    out = ""
    prev: _Line | None = None
    for ln in lines:
        text = ln.marked if use_markers else ln.plain
        if prev is None:
            out = text
        else:
            gap = ln.y0 - prev.y1
            same_page = ln.page == prev.page
            if same_page and gap > 0.8 * (prev.y1 - prev.y0):
                out += "\n" + text
            elif re.search(r"[^\s-]-$", prev.plain):
                out += text  # hyphen glued to a word: "stand-" + "alone" -> "stand-alone"
            else:
                out += " " + text
        prev = ln
    if use_markers:
        out = re.sub(r"\*\*(\s*)\*\*", r"\1", out)  # merge adjacent runs
    return re.sub(r"[ \t]+", " ", out).strip()


def _finalize(b: _Builder) -> Amendment:
    header = [ln.plain for ln in b.header]
    label = b.left_label
    if label and label in header:
        idx = header.index(label)
        author_lines, location_lines = b.header[:idx], b.header[idx + 1 :]
    elif len(b.header) >= 3:
        # Unknown layout: assume "authors..., document type, location".
        author_lines, location_lines = b.header[:-2], b.header[-1:]
        label = b.header[-2].plain
    else:
        author_lines, location_lines = b.header, []
    left = _join(b.left)
    return Amendment(
        number=b.number,
        authors=_join(author_lines, use_markers=False),
        doc_type=label,
        location=_join(location_lines, use_markers=False),
        original_text=left,
        amendment_text=_join(b.right),
        justification=_join(b.just, use_markers=False),
        language=b.language,
        page=b.page,
        is_new=not left,
    )


def parse_pdf(path: str, max_pages: int | None = None) -> ParseResult:
    """Parse an EP amendment PDF into structured amendments, with sanity warnings."""
    doc = pymupdf.open(path)
    n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
    warnings: list[str] = []

    # The cover page states the range ("AMENDMENTS 1 - 317"): used to validate the result.
    expected: tuple[int, int] | None = None
    for p in range(min(3, len(doc))):
        m = COVER_RANGE_RE.search(doc[p].get_text())
        if m:
            expected = (int(m.group(1)), int(m.group(2)))
            break

    amendments: list[Amendment] = []
    cur: _Builder | None = None
    state = IDLE

    def close(note: str | None = None) -> None:
        nonlocal cur, state
        if cur is not None:
            if note:
                warnings.append(note)
            amendments.append(_finalize(cur))
        cur, state = None, IDLE

    for page_no in range(n_pages):
        page = doc[page_no]
        mid = page.rect.width / 2
        for line in _page_lines(page, page_no + 1):
            m = AMENDMENT_RE.match(line.plain)
            if m and line.bold and line.x0 < mid:
                num = int(m.group(1))
                if state in (IDLE, HEADER) or (cur is not None and num == cur.number + 1):
                    if state != IDLE:
                        close(f"Amendment {cur.number if cur else '?'} had no end marker.")
                    cur = _Builder(number=num, page=page_no + 1)
                    state = HEADER
                    continue

            if state == IDLE:
                continue
            assert cur is not None

            if state == HEADER:
                if line.plain == "Amendment" and line.x0 >= mid and line.italic:
                    last = cur.header[-1] if cur.header else None
                    if last is not None and last.italic and abs(last.y0 - line.y0) < 4:
                        cur.left_label = cur.header.pop().plain
                    state = TABLE
                else:
                    cur.header.append(line)
                continue

            end = END_RE.match(line.plain)
            if end:
                cur.language = end.group(1).lower()
                close()
            elif state == TABLE and line.plain == "Justification" and line.bold and line.x0 < mid:
                state = JUSTIFICATION
            elif state == JUSTIFICATION:
                cur.just.append(line)
            else:
                (cur.right if line.x0 >= mid else cur.left).append(line)

    if cur is not None:
        close(f"Amendment {cur.number} had no end marker.")

    if not amendments:
        warnings.append("No amendments found. Is this an EP amendment PDF in the standard layout?")
    elif expected and max_pages is None:
        found = {a.number for a in amendments}
        missing = sorted(set(range(expected[0], expected[1] + 1)) - found)
        if missing:
            shown = ", ".join(map(str, missing[:10])) + ("..." if len(missing) > 10 else "")
            warnings.append(
                f"The cover announces amendments {expected[0]}-{expected[1]} but "
                f"{len(missing)} were not found: {shown}."
            )
    return ParseResult(amendments=amendments, warnings=warnings, expected_range=expected)
