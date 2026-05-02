import json
import pymupdf
from .parser import get_spans_from_page, group_spans_into_lines, group_lines_into_paragraphs
from .classifier import classify_paragraphs

def export_to_json(doc, output_path):

    last_chapter    = None
    last_section    = None
    last_subsection = None

    pages_output = []

    for page_number, page in enumerate(doc):

        spans      = get_spans_from_page(page)
        lines      = group_spans_into_lines(spans)
        paragraphs = group_lines_into_paragraphs(lines)
        classify_paragraphs(paragraphs)

        page_paragraphs = []

        for p in paragraphs:
            text = p.text.strip()

            if p.type == "chapter":
                last_chapter    = text
                last_section    = None
                last_subsection = None

            elif p.type == "section":
                last_section    = text
                last_subsection = None

            elif p.type == "subsection":
                last_subsection = text

            page_paragraphs.append({
                "chapter"    : last_chapter,
                "section"    : last_section,
                "subsection" : last_subsection,
                "type"       : p.type,
                "paragraph"  : " ".join(line.strip() for line in text.split("\n") if line.strip()),
            })

        if page_paragraphs:
            pages_output.append({
                "page"       : page_number + 1,
                "paragraphs" : page_paragraphs,
            })

    result = {"pages": pages_output}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"exported {len(pages_output)} pages to {output_path}")
    return result


# --- run ---
doc = pymupdf.open('/home/ahmed-walled/translateWithContext/backend/BookParsing/BooksToTry/dokumen.pub_algorithms-in-c-part-5-graph-algorithms-3rd-ed-0201361183-0785342361186-9780201361186.pdf')
export_to_json(doc, "AlgoBook.json")