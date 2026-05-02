import pymupdf
from .objects import Span, Line, Paragraph




def get_spans_from_page(page):
    spans = []

    for block in page.get_text("dict")["blocks"]:
        #only text blocks 
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                spans.append(Span(
                    text = span["text"],
                    size = span["size"],
                    font = span["font"],
                    origin=span["origin"],
                    bbox = span["bbox"],
                ))

    # smaller y0 = higher on page in PyMuPDF coordinate space
    spans.sort(key=lambda s: (s.bbox[1] , s.bbox[0]))
    return spans

def group_spans_into_lines(spans):
    lines = []
    current_spans = []
    current_center = None

    for span in spans:
        if current_center is None:
            current_spans.append(span)
            current_center = span.center_y
            continue
        #dynamic line grouping hurestic 
        #get the span height make the tolarance dynamic if the difference between y.center < thier heihgt must be in same line 
        tolerance = span.height * 0.6

        if abs(span.center_y - current_center) <= tolerance:
            current_spans.append(span)
            current_center = sum(s.center_y for s in current_spans) / len(current_spans)#update the center of the line 
        else:
            #sort depednging on x0 (spans from left to right )
            lines.append(Line(sorted(current_spans, key=lambda s: s.bbox[0])))
            current_spans = [span]
            current_center = span.center_y

    if current_spans:
        lines.append(Line(sorted(current_spans, key=lambda s: s.bbox[0])))

    return lines

def group_lines_into_paragraphs(lines):
    # smaller y0 = higher on page (importand)
    lines.sort(key=lambda l: (l.y0, l.x0))

    paragraphs = []
    current_lines = []
    prev_line = None

    for line in lines:
        if prev_line is None:
            prev_line = line
            current_lines.append(line)
            continue
        #get hight of 2 line 
        #if the vertical gap between two line << thier hieght so thier must be in same paragraph
        #dynamic thre shold adapt to the size do the font of the book text 
        avg_height     = (prev_line.height + line.height) / 2
        vertical_gap   = line.y0 - prev_line.y1      # positive when there's a gap between lines
        indent         = avg_height * 0.8

        normal_spacing = vertical_gap <= avg_height * 0.3 #move spacing from 0.8 to 0.2 to make it more tight 
        shifted_right  = line.x0 > prev_line.x0 + indent #this handle the case of repereseting a lot of paragraph in text books 
                                                    #line1 
                                                    #line2
                                                    #   line1(new paragraph)
                                                    #line
        #the only case when it is not new paragraph when no shift and normal_spacing is true 
        # rather than that it is new para 
        new_paragraph  = not normal_spacing or shifted_right 

        if new_paragraph:
            #creat new para
            paragraphs.append(Paragraph(current_lines))
            current_lines = [line]
        else:
            current_lines.append(line)

        prev_line = line

    if current_lines:
        paragraphs.append(Paragraph(current_lines))

    return paragraphs


