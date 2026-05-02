from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class Retriever:
    def __init__(self, summaries_path: str, vectorstore_dir: str = "vectorstores"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.summaries_path = summaries_path
        self.vectorstore_dir = vectorstore_dir
        self.vector_store = None

    def _load_vector_store(self, chapter_id: int):
        path = f"{self.vectorstore_dir}/chapter_{chapter_id}"

        try:
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except:
            return False

    def get_context(self, sentence: str, chapter_id: int, k: int = 5):
        """
        Inputs:
        - sentence: sentence you want context for
        - chapter_id: chapter number (e.g. 1)
        - k: number of similar sentences
        """
        
        loaded = self._load_vector_store(chapter_id)

        if not loaded:
            print(f"No vector store found for chapter {chapter_id}")
            return []

        results = self.vector_store.similarity_search(sentence, k=k)

        return [r.page_content for r in results]


if __name__ == "__main__":
    retriever = Retriever("summary.json")

    query_sentence = "The transport layer has to provide a multiplexing/demultiplexing service in order to pass data between the network layer and the correct application-level process."
    chapter_id = 3

    context = retriever.get_context(query_sentence, chapter_id, k=10)

    print("Context:\n")
    for c in context:
        print("-", c)