"""Variant C: enqueue to Celery, parse in a separate worker process (own interpreter, own GIL).

The endpoint itself only saves the upload and enqueues a task -- true
end-to-end latency (what the load driver measures) is enqueue time plus
polling /upload_pdf/status/{task_id} until the worker pool finishes parsing.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from celery.result import AsyncResult

from bench_common import MAX_UPLOAD_BYTES, unique_paths
from celery_config import celery_app
from tasks import parse_pdf_task

app = FastAPI(title="Benchmark Variant C - celery")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_pdf", status_code=202)
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

    task = parse_pdf_task.delay(str(upload_path), str(output_path))

    return {
        "message": "PDF uploaded. Parsing has been queued.",
        "task_id": task.id,
        "task_status": "PENDING",
    }


@app.get("/upload_pdf/status/{task_id}")
def upload_pdf_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": task.state}
    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.result)
    return response
