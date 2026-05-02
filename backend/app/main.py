import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from Search.basicSearch import search_for_text
from context.ContexBasicHandling import get_context
from model.translator_pro import translate_function
from parser.exporter import export_to_json
import pymupdf

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

# 🔹 Load once (NOT per request)
with BOOK_PATH.open('r', encoding='utf-8') as f:
    BOOK_DATA = json.load(f)

class Translate_Req(BaseModel):
    text: str
    page_number: int


@app.get("/")
def root():
    """Health check and welcome endpoint"""
    return {
        "message": "Context-Aware Translation API",
        "status": "running",
        "docs": "/docs"
    }

 

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF, parse it, and generate output.json"""
    try:
        # Validate file type
        if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Check file size (max 50MB)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Parse PDF and generate JSON
        doc = pymupdf.open(str(file_path))
        parsed_data = export_to_json(doc, str(BOOK_PATH))
        doc.close()
        
        # Reload BOOK_DATA with new content
        global BOOK_DATA
        with BOOK_PATH.open('r', encoding='utf-8') as f:
            BOOK_DATA = json.load(f)
        
        return {
            "message": "PDF uploaded and parsed successfully",
            "filename": file.filename,
            "file_path": str(file_path),
            "pages_parsed": len(parsed_data.get("pages", [])),
            "output_json": str(BOOK_PATH)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

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
    context_paragraph , target_text = get_context(search_output=searched_text ,book_obj=BOOK_DATA,target_text=text )
    translation = translate_function(target_text,context_paragraph)

    return {
        "translation": translation
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)