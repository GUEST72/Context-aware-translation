import pymupdf
from objects import Span, Line, Paragraph
from parser import get_spans_from_page , group_spans_into_lines , group_lines_into_paragraphs

import re


def uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0
    upper = [c for c in letters if c.isupper()]
    return len(upper) / len(letters)

def classify_paragraphs(paragraphs):

    # ---- compute page average font size ----
    sizes = [p.avg_size for p in paragraphs if p.avg_size]
    page_avg_size = sum(sizes) / len(sizes) if sizes else 0

    chapter_threshold = page_avg_size * 1.4
    section_threshold = page_avg_size * 1.2

    # ---- chapter regex patterns ----
    chapter_patterns = [
        r"^chapter\s+\d+",
        r"^chapter\s+\d+\s*[:\-–•]",
        r"^\d+\s*chapter",
        r"^chapter\s+[ivxlcdm]+"
    ]

    for p in paragraphs:

        text       = p.text.strip()
        text_lower = text.lower()
        p.type     = "body"

        # only classify headings if single line
        if len(p.lines) != 1:
            continue

        # ---- section numbering detection ----
        if re.match(r"^\d+\.\d+\.\d+", text):
            p.type = "subsection"
            continue

        if re.match(r"^\d+\.\d+", text):
            p.type = "section"
            continue

        # ---- chapter keyword ----
        if "chapter" in text_lower:
            p.type = "chapter"
            continue

        # ---- chapter regex patterns ----
        for pattern in chapter_patterns:
            if re.match(pattern, text_lower):
                p.type = "chapter"
                break

        if p.type == "chapter":
            continue

        # ---- uppercase heuristic ----
        if uppercase_ratio(text) > 0.6:
            p.type = "section"
            continue

        # ---- dynamic font size heuristic ----
        if p.avg_size > chapter_threshold:
            p.type = "chapter"
            continue

        if p.avg_size > section_threshold:
            p.type = "section"
            continue

    return paragraphs
# --- run ---
doc = pymupdf.open('/home/ahmed-walled/translateWithContext/backend/BookParsing/BooksToTry/computer-networking-a-top-down-approach-8th-edition.pdf')

'''
page       = doc[14]
blocks = page.get_text("blocks" , sort=True)
for block in blocks :
    print(block , "\n");
'''

for page in doc[12:13]:
    spans      = get_spans_from_page(page)
    lines      = group_spans_into_lines(spans)
    paragraphs = group_lines_into_paragraphs(lines)
    classify_paragraphs(paragraphs)
    
    for para in paragraphs :
        print(para)
        print("\n");