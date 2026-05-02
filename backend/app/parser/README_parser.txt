README - Parser Pipeline

Overview
- Library-light, heuristic-driven PDF → structured JSON pipeline implemented with PyMuPDF.
- Converts book PDFs into a document model of Spans → Lines → Paragraphs, classifies headings (chapter/section/subsection), and exports a page-wise JSON with chapter/section context for each paragraph.
- Primary files: parser.py (grouping pipeline), objects.py (Span/Line/Paragraph data classes), classifier.py (heading heuristics), exporter.py (JSON exporter), main.py (entrypoint placeholder).

Pipeline steps
1. Text extraction (parser.get_spans_from_page)
   - Uses PyMuPDF page.get_text("dict") to iterate text blocks and spans.
   - Creates Span objects capturing text, font size, font name, origin (baseline), and bbox (x0,y0,x1,y1).
   - Spans are sorted top→bottom then left→right.

2. Line grouping (parser.group_spans_into_lines)
   - Groups nearby spans into lines using the span origin (baseline) and a dynamic vertical tolerance.
   - Tolerance scales with span.height (span.height * 0.6) so the grouping adapts to font sizes and avoids jitter across different pages.
   - Spans within tolerance are merged; each line becomes a Line object with bounding box and average font size.

3. Paragraph segmentation (parser.group_lines_into_paragraphs)
   - Lines are sorted by y0 and x0.
   - Paragraph breaks are detected using a combination of vertical gap vs average line height and indentation heuristics:
     - Compute avg_height between consecutive lines and vertical_gap = line.y0 - prev_line.y1.
     - If vertical_gap is larger than an adaptive threshold or the line is shifted right beyond indent (avg_height * 0.8), start a new paragraph.
   - Produces Paragraph objects with bounding box, avg_size and concatenated text.

4. Heading classification (classifier.classify_paragraphs)
   - Computes page average font size and derives size thresholds:
     - chapter_threshold = page_avg_size * 1.4
     - section_threshold = page_avg_size * 1.2
   - Uses multiple heuristics to label single-line paragraphs as chapter / section / subsection:
     - Numeric numbering patterns (e.g., ^\d+\.\d+)
     - Chapter keywords and regexes (e.g., "chapter \d+", roman numerals)
     - Uppercase ratio heuristic (uppercase letters fraction > 0.6 → likely heading)
     - Font size compared against dynamic thresholds
   - Falls back to "body" for regular paragraphs.

5. Export (exporter.export_to_json)
   - Iterates pages, applies the pipeline, and maintains last_chapter/last_section/last_subsection state to attach hierarchical context to each paragraph.
   - Exports JSON shaped as:
     {
       "pages": [
         {
           "page": <int>,
           "paragraphs": [
             {
               "chapter": <string|null>,
               "section": <string|null>,
               "subsection": <string|null>,
               "type": "chapter"|"section"|"subsection"|"body",
               "paragraph": "<cleaned paragraph text>"
             },
             ...
           ]
         },
         ...
       ]
     }
   - Example output: backend/app/output.json

How to run (example)
- Python REPL / script:
  from pathlib import Path
  import pymupdf
  from parser.exporter import export_to_json
  doc = pymupdf.open("path/to/book.pdf")
  export_to_json(doc, "backend/app/output.json")

Design highlights
- Minimal 3rd-party logic: relies mainly on PyMuPDF for low-level extraction; all grouping/classification logic is custom heuristics for better interpretability and robustness on technical books.
- Dynamic thresholds adapt to page-level typography (font sizes, spacing) improving generalization across varied layouts.
- Uses baseline origin (span.origin) to reduce line-jitter issues when text rendering boxes vary.

Limitations & edge cases
- Multi-column layouts, complex multi-line headings, heavy inline figures/tables, and decorative fonts may reduce accuracy.
- Does not perform OCR — input PDFs must contain embedded text.
- Current heuristics assume mostly single-column, typographic textbooks; exotic layouts may need extra rules.

Suggested improvements
- Add column-detection and column-aware grouping.
- Integrate a small layout classifier to pre-detect two-column vs single-column pages.
- Use an ML-based boundary detector (train on labeled paragraphs) for higher accuracy on ambiguous cases.
- Add unit tests using a small corpus of labeled pages and a CI check to guard regressions.

Notes
- File references: backend/app/parser/parser.py, objects.py, classifier.py, exporter.py, main.py
