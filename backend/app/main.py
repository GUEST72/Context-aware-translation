import json
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from Search.basicSearch import search_for_text
from context.ContexBasicHandling import get_context

app = FastAPI()

# 🔹 Load once (NOT per request)
with open('/home/ahmed-walled/Projects/Context-Aware-Translation/backend/app/output.json', 'r') as f:
    BOOK_DATA = json.load(f)

class Translate_Req(BaseModel):
    text: str
    page_number: int


@app.post("/Translate")
def translate(text_to_trans: Translate_Req):
    text = text_to_trans.text
    page_number = text_to_trans.page_number
    text = text.replace("\n", " ")
    print(text)
    searched_text = search_for_text(
        book_Jason='/home/ahmed-walled/Projects/Context-Aware-Translation/backend/app/output.json',
        text=text,
        page_number=page_number
    )

    if searched_text is None:
        return {"error": "Text not found"}
    context_paragraph , target_text = get_context(search_output=searched_text ,book_obj=BOOK_DATA,target_text=text )

    return {
        "tatget_text": target_text,
        "context_paragraph": context_paragraph,
        "match_type": searched_text['match_type']
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)