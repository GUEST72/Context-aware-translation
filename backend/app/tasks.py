from pathlib import Path

import pymupdf

from celery_config import celery_app
from parser.exporter import export_to_json


@celery_app.task(name="parse_pdf_task")
def parse_pdf_task(file_path: str, output_path: str) -> dict:
    file_path_obj = Path(file_path)
    output_path_obj = Path(output_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"Uploaded file not found: {file_path_obj}")

    doc = pymupdf.open(str(file_path_obj))
    try:
        parsed_data = export_to_json(doc, str(output_path_obj))
    finally:
        doc.close()

    pages_parsed = len(parsed_data.get("pages", []))
    return {
        "message": "PDF parsed successfully",
        "filename": file_path_obj.name,
        "pages_parsed": pages_parsed,
        "output_json": str(output_path_obj),
    }
