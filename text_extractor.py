import json
import re


class TextExtractor:
    CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+\d+$")
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def extract_chapters(self):
        chapters = []
        current = None
        expect_title = False

        for page in self.data.get("pages", []):
            for p in page.get("paragraphs", []):
                text = p.get("paragraph", "").strip()
                p_type = p.get("type")

                if not text:
                    continue

                if p_type == "chapter" and self.CHAPTER_PATTERN.match(text):
                    if current:
                        chapters.append(current)

                    current = {
                        "chapter_id": text,
                        "title": "",
                        "content": []
                    }
                    expect_title = True
                    continue

                if expect_title and p_type == "body":
                    current["title"] = text
                    expect_title = False
                    continue

                if current and p_type == "body" and len(text) > 40:
                    text = text.replace("\n", " ")

                    text = re.sub(r"\s+", " ", text)

                    current["content"].append(text)

        if current:
            chapters.append(current)

        return chapters

    def split_chapter_into_sentences(self, chapter):
        sentences = []

        for paragraph in chapter.get("content", []):
            for s in self.SENTENCE_PATTERN.split(paragraph):
                s = s.strip()

                if len(s) < 40:
                    continue

                if "Figure" in s or "♦" in s:
                    continue

                sentences.append(s)

        return sentences


if __name__ == "__main__":
    extractor = TextExtractor("output.json")

    chapters = extractor.extract_chapters()
    print(f"Chapters found: {len(chapters)}")

    if not chapters:
        print("No chapters found.")
    else:
        first = chapters[0]

        print("\n--- First Chapter ---")
        print("ID:", first["chapter_id"])
        print("Title:", first["title"])
        print("Paragraphs:", len(first["content"]))

        sentences = extractor.split_chapter_into_sentences(first)
        print("Sentences:", len(sentences))

        print("\nSample sentences:")
        for s in sentences[:5]:
            print("-", s)