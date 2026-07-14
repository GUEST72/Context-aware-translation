"""Variant A: async def upload_pdf, parsing runs inline on the event-loop thread.

This reproduces the current production shape of a CPU-bound handler declared
`async def` with no offloading: nothing here yields control back to the
event loop during the pymupdf.open()/export_to_json() call, so a single
in-flight parse blocks every other coroutine (including /health) on this
worker.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException

from bench_common import MAX_UPLOAD_BYTES, parse_pdf, unique_paths

app = FastAPI(title="Benchmark Variant A - async inline")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    upload_path, output_path = unique_paths(file.filename)
    with open(upload_path, "wb") as f:
        f.write(content)

    parsed = parse_pdf(upload_path, output_path)

    return {
        "message": "PDF parsed",
        "pages_parsed": len(parsed.get("pages", [])),
        "output_json": str(output_path),
    }
