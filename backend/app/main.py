import json
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from celery.result import AsyncResult
from pydantic import BaseModel
from Search.basicSearch import search_for_text
from context.ContexBasicHandling import get_context
from model.translator_pro import translate_function
from celery_config import celery_app
from tasks import parse_pdf_task

app = FastAPI(title="Context-Aware Translation API", description="Translate text with contextual awareness")

# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
BOOK_PATH = BASE_DIR / "output.json"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

class Translate_Req(BaseModel):
    text: str
    page_number: int


def _load_book_data() -> dict:
    if not BOOK_PATH.exists():
        raise HTTPException(status_code=503, detail="Parsed book data is not available yet")
    with BOOK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root():
    """Health check and welcome endpoint"""
    return {
        "message": "Context-Aware Translation API",
        "status": "running",
        "docs": "/docs"
    }

 

@app.post("/upload_pdf", status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF, enqueue parse task, and return task metadata."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing file name")

        # Validate file type
        if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Check file size (max 50MB)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        # Save uploaded file
        original_filename = Path(file.filename).name
        suffix = Path(original_filename).suffix or ".pdf"
        stored_filename = f"{uuid.uuid4().hex}{suffix}"
        file_path = UPLOAD_DIR / stored_filename
        with open(file_path, "wb") as f:
            f.write(content)

        task = parse_pdf_task.delay(str(file_path), str(BOOK_PATH))

        return {
            "message": "PDF uploaded. Parsing has been queued.",
            "filename": original_filename,
            "stored_filename": stored_filename,
            "file_path": str(file_path),
            "task_id": task.id,
            "task_status": "PENDING",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/upload_pdf/status/{task_id}")
def upload_pdf_status(task_id: str):
    """Check parsing task state and result."""
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": task.state}

    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.result)

    return response

@app.post("/Translate")
def translate(text_to_trans: Translate_Req):
    text = text_to_trans.text
    page_number = text_to_trans.page_number
    text = text.replace("\n", " ")
    searched_text = search_for_text(
        book_Jason=str(BOOK_PATH),
        text=text,
        page_number=page_number
    )

    if searched_text is None:
        return {"error": "Text not found"}

    book_data = _load_book_data()
    context_paragraph , target_text = get_context(search_output=searched_text ,book_obj=book_data,target_text=text )
    translation = translate_function(target_text,context_paragraph)

    return {
        "translation": translation
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
