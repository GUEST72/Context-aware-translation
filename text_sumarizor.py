from sentence_transformers import SentenceTransformer
import numpy as np
import json
from text_extractor import TextExtractor
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class ExtractiveSummarizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def _embed(self, sentences: list) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def _mmr(self, embeddings: np.ndarray, doc_embedding: np.ndarray, k: int, lambda_: float):
        n = len(embeddings)

        selected = []
        candidates = set(range(n))

        sim_to_doc = embeddings @ doc_embedding

        first = np.argmax(sim_to_doc)
        selected.append(first)
        candidates.remove(first)


        max_sim_to_selected = np.zeros(n)

        while len(selected) < k and candidates:
            last_selected = selected[-1]

            sims = embeddings @ embeddings[last_selected]
            max_sim_to_selected = np.maximum(max_sim_to_selected, sims)

            scores = lambda_ * sim_to_doc - (1 - lambda_) * max_sim_to_selected

            for s in selected:
                scores[s] = -1e9

            best_idx = np.argmax(scores)

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def summarize(self, sentences: list, ratio: float = 0.25, lambda_: float = 0.85):
        """
        Generate extractive summary using embeddings + MMR
        Returns:
            summary_sentences, remaining_sentences
        """
        if not sentences:
            return [], []

        embeddings = self._embed(sentences)
        doc_embedding = np.mean(embeddings, axis=0)
        
        k = max(1, int(len(sentences) * ratio))

        selected_indices = self._mmr(embeddings, doc_embedding, k, lambda_)
        selected_indices = sorted(selected_indices)

        selected = [sentences[i] for i in selected_indices]
        return selected


def summarize_book(input_path: str, output_path: str, ratio=0.25, lambda_=0.85):
    extractor = TextExtractor(input_path)
    summarizer = ExtractiveSummarizer()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    chapters = extractor.extract_chapters()
    results = []

    os.makedirs("vectorstores", exist_ok=True)

    for chapter in chapters:
        sentences = extractor.split_chapter_into_sentences(chapter)
        summary = summarizer.summarize(sentences, ratio, lambda_)

        chapter_num = chapter["chapter_id"].split()[-1]
        vs_path = f"vectorstores/chapter_{chapter_num}"

        vector_store = FAISS.from_texts(summary, embeddings)
        vector_store.save_local(vs_path)

        results.append({
            "chapter_id": chapter["chapter_id"],
            "title": chapter["title"],
            "summary": summary
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
        
if __name__ == "__main__":
    summarize_book("output.json", "summary.json", ratio=0.25, lambda_=0.85)