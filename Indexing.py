import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(type(data))
    return data


def convert_to_documents(data):
    docs = []

    for page in data["pages"]:
        page_num = page["page"]

        for p in page["paragraphs"]:
            docs.append(
                Document(
                    page_content=p["paragraph"],
                    metadata={
                        "page": page_num,
                        "chapter": p.get("chapter"),
                        "section": p.get("section"),
                        "subsection": p.get("subsection"),
                        "type": p.get("type")
                    }
                )
            )
    return docs


def clean_documents(docs):
    clean_docs = []
    buffer_doc = None 

    for doc in docs:
        text = doc.page_content.strip()
        lower_text = text.lower()

        if len(text) < 50:
            continue

        if "contents" in lower_text or "index" in lower_text:
            continue

        should_merge = (
            buffer_doc is not None and
            len(text) < 500 and
            len(buffer_doc.page_content) < 2000 and   
            doc.metadata.get("chapter") == buffer_doc.metadata.get("chapter") and
            doc.metadata.get("section") == buffer_doc.metadata.get("section") 
        )

        if should_merge:
            buffer_doc = Document(
                page_content=buffer_doc.page_content + "\n\n" + text,
                metadata=buffer_doc.metadata
            )
            continue

        if buffer_doc is not None:
            clean_docs.append(buffer_doc)

        buffer_doc = doc
        
    if buffer_doc is not None:
        clean_docs.append(buffer_doc)

    return clean_docs


def split_documents(clean_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    final_docs = []

    for doc in clean_docs:
        if len(doc.page_content) > 1000:
            split_docs = splitter.split_documents([doc])
            final_docs.extend(split_docs)
        else:
            final_docs.append(doc)

    return final_docs


def get_document_chunks(file_name):
    data = load_json(file_name)
    documents = convert_to_documents(data)
    clean_docs = clean_documents(documents)
    final_docs = split_documents(clean_docs)
    return final_docs


def create_vector_store(documents, name='vector_store'):
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(name)
    print("Vector store saved successfully")
    return 


def preprocess_and_store(file_name, output='vector_store'):
    documents = get_document_chunks(file_name)
    create_vector_store(documents, output)
    



if __name__ == "__main__":

    data = load_json("output.json")
    documents = convert_to_documents(data)
    print(f"Number of documents: {len(documents)}")

    clean_docs = clean_documents(documents)
    print(f"First document: {clean_docs[0].page_content}")
    print(f"Number of clean documents: {len(clean_docs)}")

    lengths = [len(doc.page_content) for doc in clean_docs]
    print("Min:", min(lengths))
    print("Max:", max(lengths))
    print("Avg:", sum(lengths)//len(lengths))
    
    final_docs = split_documents(clean_docs)
    print(f"Number of final documents: {len(final_docs)}")

    final_lengths = [len(doc.page_content) for doc in final_docs]
    print("Min:", min(final_lengths))
    print("Max:", max(final_lengths))
    print("Avg:", sum(final_lengths)//len(final_lengths))

    preprocess_and_store("output.json")


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True  
    )

    query = "What is the purpose of TCP?"

    results = vector_store.similarity_search(query, k=5)

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(result.page_content)
        print("Metadata:", result.metadata)
        print("-" * 50)