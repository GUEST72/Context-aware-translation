"""Variant B: plain `def` upload_pdf, offloaded by Starlette to its AnyIO threadpool.

Identical logic to variant A, but declaring the path operation as a
synchronous function makes FastAPI run it via `anyio.to_thread.run_sync`
instead of on the event-loop thread, so the event loop (and /health) stays
responsive while parsing runs on a worker thread. `await file.read()` is
replaced with the sync equivalent (`file.file.read()`) since this function
can't await.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException

from bench_common import MAX_UPLOAD_BYTES, parse_pdf, unique_paths

app = FastAPI(title="Benchmark Variant B - threadpool")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = file.file.read()
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
