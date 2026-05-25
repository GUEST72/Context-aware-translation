"""
Quick demo: extract terms + definitions and display them in a readable format.
Run from the backend/ directory:  python demo_terminology.py
"""
import json, sys, textwrap, time
sys.path.insert(0, ".")
from app.terminology import TerminologyExtractor, DefinitionExtractor

BAR_WIDTH = 20
SEP       = "─" * 70

def bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    return "|" + "█" * filled + "░" * (BAR_WIDTH - filled) + "|"

def wrap(text: str, width: int = 60, indent: str = "    ") -> str:
    lines = textwrap.wrap(text, width)
    return "\n".join(indent + l for l in lines)

def main():
    with open("app/output.json") as f:
        book = json.load(f)

    # ── Extraction ──────────────────────────────────────────────────────────
    t0 = time.time()
    te = TerminologyExtractor()
    terms = te.extract(book)
    print(f"Terms: {len(terms)}  ({time.time()-t0:.1f}s)")

    # ── Definition space ────────────────────────────────────────────────────
    de = DefinitionExtractor()
    t1 = time.time()
    space = de.build_space(book, [t.term for t in terms])
    print(f"Space: {len(space)}/{len(terms)} terms  ({time.time()-t1:.1f}s)\n")

    # ── Display ─────────────────────────────────────────────────────────────
    # Show phrases first (sorted by confidence desc), then single tokens
    phrases  = [t for t in terms if " " in t.term]
    singles  = [t for t in terms if " " not in t.term]

    # pick top-N by weirdness, then sort results by confidence
    candidates = phrases[:60] + singles[:60]
    results = []
    for t in candidates:
        res = de.query(t.term, space)
        if res:
            results.append(res)
    results.sort(key=lambda r: r.confidence, reverse=True)

    for res in results:
        conf_str = f"{res.confidence:.2f}"
        header = f"  {res.term}  (p.{res.source_page})  {conf_str} {bar(res.confidence)}"
        print(header)
        print(wrap(res.definition))
        print(SEP)
        print()

if __name__ == "__main__":
    main()
