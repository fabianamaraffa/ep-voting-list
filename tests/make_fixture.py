"""Builds a small PDF that imitates the layout of an EP amendments document.

It reproduces the awkward cases of the real format: running headers and footers, the
two-column table with italic labels, bold-italic changes, a heading separated from its
authors by a page break, a table split across pages, a hyphenated line wrap, a
justification block, and an in-text mention of "Amendment 5" that must not split anything.
"""

from __future__ import annotations

from itertools import zip_longest

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

W, H = A4
LEFT_X, RIGHT_X = 70, 314
REG, BOLD, ITAL, BI = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"
SIZE = 11
LEADING = 14


class _Doc:
    def __init__(self, path: str):
        self.c = canvas.Canvas(path, pagesize=A4)
        self.page = 1
        self.y = H - 90

    def footer(self):
        c = self.c
        c.setFont(REG, 9)
        c.drawString(70, 60, "PE766.605v02-00")
        c.drawString(W / 2 - 15, 60, f"{self.page}/6")
        c.drawRightString(W - 70, 60, "AM\\1310525EN.docx")
        c.setFont(BOLD, 18)
        c.drawString(40, 28, "EN")

    def new_page(self):
        self.footer()
        self.c.showPage()
        self.page += 1
        self.y = H - 90

    def runs(self, x, runs, y=None):
        cx = x
        for text, font in runs:
            self.c.setFont(font, SIZE)
            self.c.drawString(cx, y if y is not None else self.y, text)
            cx += stringWidth(text, font, SIZE)

    def line(self, text, font=REG, x=LEFT_X):
        self.runs(x, [(text, font)])
        self.y -= LEADING

    def gap(self, n=1):
        self.y -= LEADING * n

    def heading(self, number, authors, doc_type, location):
        self.line(f"Amendment  {number}", BOLD)
        for a in authors:
            self.line(a, BOLD)
        self.gap()
        self.line(doc_type, BOLD)
        self.line(location, BOLD)
        self.gap()
        self.runs(120, [(doc_type, ITAL)])
        self.runs(391, [("Amendment", ITAL)])
        self.y -= LEADING * 1.6

    def table(self, left, right, break_after=None):
        for i, (lt, rt) in enumerate(zip_longest(left, right)):
            if lt:
                self.runs(LEFT_X, lt)
            if rt:
                self.runs(RIGHT_X, rt)
            self.y -= LEADING
            if break_after is not None and i == break_after:
                self.new_page()

    def end(self, lang="en"):
        self.gap()
        self.c.setFont(REG, SIZE)
        self.c.drawRightString(W - 100, self.y, f"Or. {lang}")
        self.gap(2)

    def save(self):
        self.footer()
        self.c.save()


def build(path: str) -> None:
    """Write the fixture PDF (6 amendments, 6 pages)."""
    d = _Doc(path)
    c = d.c
    c.setFont(BOLD, 28)
    c.drawString(140, H - 300, "AMENDMENTS")
    c.drawString(140, H - 335, "1 - 6")
    d.footer()
    c.showPage()
    d.page = 2
    d.footer()
    c.showPage()
    d.page = 3
    d.y = H - 90

    d.heading(1, ["Anna Example, Bruno Sample"], "Motion for a resolution", "Citation 3 a (new)")
    d.table([], [[("– having regard to the Youth Guarantee,", BI)]])
    d.end()

    d.heading(2, ["Carla Test"], "Motion for a resolution", "Paragraph 3")
    d.table(
        [[("3. Calls for support ", REG), ("existing", BI), (" instruments;", REG)]],
        [[("3. Calls for support ", REG), ("new youth", BI), (" instruments;", REG)]],
    )
    d.end()

    # AM 3: only the "Amendment 3" line fits on this page; its authors are on the next.
    d.y = 150
    d.line("Amendment  3", BOLD)
    d.new_page()
    d.line("Dora Mid, Emil Long-", BOLD)
    d.line("Name Third", BOLD)
    d.gap()
    d.line("Motion for a resolution", BOLD)
    d.line("Paragraph 5", BOLD)
    d.gap()
    d.runs(120, [("Motion for a resolution", ITAL)])
    d.runs(391, [("Amendment", ITAL)])
    d.y -= LEADING * 1.6
    d.table(
        [[("5. Stresses the stand-", REG)], [("alone nature of the fund;", REG)]],
        [[("5. Stresses the ", REG), ("independent", BI)], [("nature of the fund;", REG)]],
    )
    d.end("de")

    # AM 4: table split across a page break, plus a justification
    d.heading(4, ["Fiona Four"], "Motion for a resolution", "Paragraph 6")
    d.table(
        [[("6. First line of text", REG)], [("6. second line of text", REG)], [("third line of text;", REG)]],
        [[("6. First line of text", REG)], [("second line ", REG), ("with change", BI)], [("third line of text;", REG)]],
        break_after=0,
    )
    d.gap()
    d.line("Justification", BOLD)
    d.line("Funding must stay with youth organisations.", REG)
    d.end()

    # AM 5: the text mentions "Amendment 5" on its own line inside the table
    d.heading(5, ["Gus Five"], "Motion for a resolution", "Paragraph 7")
    d.table(
        [[("7. Notes the position", REG)], [("Amendment 5", REG)], [("of the committee;", REG)]],
        [[("7. Notes the position", REG)], [("Amendment 5", REG)], [("of the ", REG), ("whole", BI), (" committee;", REG)]],
    )
    d.end()

    # AM 6: deletion
    d.heading(6, ["Hana Six"], "Motion for a resolution", "Paragraph 8")
    d.table([[("8. Text to be removed;", REG)]], [[("deleted", REG)]])
    d.end("hu")
    d.save()


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "fixture.pdf")
