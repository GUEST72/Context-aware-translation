import json
from pathlib import Path
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from Search.basicSearch import search_for_text
from context.ContexBasicHandling import get_context
from model.translator_pro import translate_function

app = FastAPI(title="Context-Aware Translation API", description="Translate text with contextual awareness")
BASE_DIR = Path(__file__).resolve().parent
BOOK_PATH = BASE_DIR / "output.json"

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


@app.post("/uploud_book")
def uploud_book():
     pass

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