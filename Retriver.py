from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class Retriever():
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
    
    def load_vector_store(self, name ='vector_store'):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.load_local(
            name,
            self.embeddings,
            allow_dangerous_deserialization=True  
        )
        
    def retrieve(self, query, k=5):
        results = self.vector_store.similarity_search(query, k=k)
        content = []
        metadata = []
        for _, result in enumerate(results):
            content.append(result.page_content)
            metadata.append(result.metadata)
            
        return content, metadata
        
    
if __name__ == "__main__":
    retriver = Retriever()
    retriver.load_vector_store("vector_store")
    
    query = "What is the purpose of TCP?"
    content, metadata = retriver.retrieve(query=query, k=10)
    
    for i, (c, m) in enumerate(zip(content, metadata)):
        print(f"Result {i+1}:")
        print(c)
        print("Metadata:", m)