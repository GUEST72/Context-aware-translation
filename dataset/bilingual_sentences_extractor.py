import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import spacy
import stanza
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import argparse

# Book will be returned as a string in preparation for subsequent sentence splitting and alignment
def extract_text_from_epub(book_path):
    book = epub.read_epub(book_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator= ' ', strip=True)
            if text:
                chapters.append(text)
    return " ".join(chapters)

def segment_english_and_arabic(english_text, arabic_text):
    # English model using spaCy
    # Additional processing
    nlp_en = spacy.load('en_core_web_sm')
    doc = nlp_en(english_text)
    en_sentences = []
    for sent in doc.sents:
        word_count = len([token for token in sent if not token.is_punct and not token.is_space])
        has_verb = any(token.pos_ in ["VERB", "AUX"] for token in sent)    
        raw_text = sent.text.strip()
        alpha_ratio = sum(c.isalpha() for c in raw_text) / max(1, len(raw_text))    
        if word_count >= 2 and has_verb and alpha_ratio > 0.5:
            en_sentences.append(raw_text)

    # Arabic model using Stanza
    # Doing some preprocessing to ensure better sentence segmentation
    # No fixed boundaries
    nlp_ar = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos')
    doc_ar = nlp_ar(arabic_text)
    ar_sentences = []
    for sent in doc_ar.sentences:
        word_count = len([word for word in sent.words if not word.upos in ["PUNCT", "SYM", "X"]])
        has_verb = any(word.upos in ["VERB", "AUX"] for word in sent.words)    
        raw_text = sent.text.strip()
        alpha_ratio = sum(c.isalpha() for c in raw_text) / max(1, len(raw_text))    
        if word_count >= 2 and has_verb and alpha_ratio > 0.5:
            ar_sentences.append(raw_text)
    
    return en_sentences, ar_sentences

def embed_and_match_closest_k_pairs(en_sentences, ar_sentences, k=1):
    model = SentenceTransformer('sentence-transformers/LaBSE')
    en_embeddings = model.encode(en_sentences, convert_to_numpy=True, normalize_embeddings=True)
    ar_embeddings = model.encode(ar_sentences, convert_to_numpy=True, normalize_embeddings=True)

    # Build FAISS index for Arabic sentences
    dimension = en_embeddings.shape[1]
    # Using Inner Product (IP) for cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(ar_embeddings)

    # Search for the closest Arabic sentence for each English sentence
    top_k_neighbors = 1
    similarities, indices = index.search(en_embeddings, top_k_neighbors)

    matched_pairs = []
    for idx, english_sentence in enumerate(en_sentences):
        confidence_score_for_match = similarities[idx][0]
        # Can vary (increase) the threshold if many (false) misses occur
        if confidence_score_for_match > 0.5:
            best_arabic_index = indices[idx][0]
            matched_pairs.append({'english' : english_sentence, 'arabic' : ar_sentences[best_arabic_index], 'confidence score' : confidence_score_for_match})
    
    data_frame = pd.DataFrame(matched_pairs).sort_values(by='confidence score', ascending=False)
    # Keep only k matches for k pairs
    matched_pairs_df = data_frame.head(k)
    # Make CSV file
    matched_pairs_df.to_csv('matched_sentences.csv', index=False, encoding='utf-8-sig')
    return matched_pairs_df

if __name__ == "__main__":
    # Add arguments for book paths and for matches needed
    parser = argparse.ArgumentParser(description="Extract and match bilingual sentences from ePub books.")
    parser.add_argument("--en", help="Path to the English book (epub)")
    parser.add_argument("--ar", help="Path to the Arabic book (epub)")
    parser.add_argument("--k", type=int, default=20, help="Number of matches to find")   
    args = parser.parse_args()

    english_book_path = args.en
    arabic_book_path = args.ar
    k = args.k

    english_text = extract_text_from_epub(english_book_path)
    arabic_text = extract_text_from_epub(arabic_book_path)

    en_sentences, ar_sentences = segment_english_and_arabic(english_text, arabic_text)

    matched_pairs = embed_and_match_closest_k_pairs(en_sentences, ar_sentences, k=k)