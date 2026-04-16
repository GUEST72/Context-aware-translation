class Span:
    def __init__(self, text, size, font, origin ,bbox):
        self.text = text
        self.size = size
        self.font = font
        self.origin = origin
        self.bbox = bbox  # (x0, y0, x1, y1)
        #using bbox is much better than (x.u) bcs this will allow as to interprut properties like hieght of the line , center y ,
        #allow us to put dynamic threshold for generlazation .    
    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]
    
    #it is better to use y in origin that represent base line of the span
    #The origin is the Baseline of the text. This is the invisible line that letters "sit" on.
    #this make us avoid problem of Line Jitter (that y0 and y1 get affected by)
    @property
    def center_y(self):
        return self.origin[1]  

    
class Line:
    def __init__(self, spans):
        self.spans = spans
        self._update_metrics()

    def _update_metrics(self):
        self.bbox = (
            min(s.bbox[0] for s in self.spans),
            min(s.bbox[1] for s in self.spans),
            max(s.bbox[2] for s in self.spans),
            max(s.bbox[3] for s in self.spans),
        )
        self.avg_size = sum(s.size for s in self.spans) / len(self.spans)

    @property
    def x0(self): return self.bbox[0]

    @property
    def y0(self): return self.bbox[1]

    @property
    def x1(self): return self.bbox[2]

    @property
    def y1(self): return self.bbox[3]

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def center_y(self):
        return (self.y0 + self.y1) / 2

    @property
    def text(self):
        return " ".join(s.text.strip() for s in self.spans)

class Paragraph:
    def __init__(self , lines):
        self.lines = lines
        self.type = "body"  # default, can be "chapter" or "section"
        self._update_metrics()

    def _update_metrics(self):
        self.bbox = (
            min(line.x0 for line in self.lines) ,
            min(line.y0 for line in self.lines) ,
            max(line.x1 for line in self.lines) ,
            max(line.y1 for line in self.lines) ,
        )
        self._avg_size = sum(line.avg_size for line in self.lines) / len(self.lines)

    @property
    def x0(self): return self.bbox[0]

    @property
    def y0(self): return self.bbox[1]

    @property
    def x1(self): return self.bbox[2]

    @property
    def y1(self): return self.bbox[3]

    @property
    def avg_size(self):
        return self._avg_size

    @property
    def text(self):
        return "\n".join(l.text for l in self.lines)

    def __repr__(self):
        return f"Paragraph(avg_size={self.avg_size:.1f}, lines={len(self.lines)}, type={self.type}, text={self.text[:2000]})"