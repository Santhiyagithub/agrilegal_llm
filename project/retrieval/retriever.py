import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

class LegalRetriever:
    def __init__(self, distance_threshold=0.3):
        # In Chroma, distance is returned. If collection is configured as cosine space:
        # Distance = 1 - Cosine Similarity. 
        # A similarity threshold of 0.85 equates to a distance threshold of 0.15.
        # We set default to 0.3 (similarity 0.70) to be a bit forgiving, but can be tuned to 0.15.
        self.threshold = distance_threshold
        print("Initializing embedding model (sentence-transformers/all-mpnet-base-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            # Ensure we use cosine distance metric
            collection_metadata={"hnsw:space": "cosine"}
        )

    def retrieve_and_validate(self, query: str, top_k: int = 4):
        """
        Gate 1: Evidence Confidence Validation
        Prevents LLM invocation if retrieved evidence is weak.
        """
        try:
            # Perform similarity search with distance scores (lower is better for distance)
            results = self.db.similarity_search_with_score(query, k=top_k)
            
            if not results:
                return False, "Insufficient authoritative information available."

            # best_score is the lowest distance
            best_doc, best_score = results[0]
            
            # Since distance means "lower is closer", we reject if best score is strictly greater than threshold
            if best_score > self.threshold:
                print(f"[Gate 1 Rejection] Best match distance ({best_score:.4f}) Exceeds threshold ({self.threshold})")
                return False, "Insufficient authoritative information available."
            
            print(f"[Gate 1 Passed] Best match distance: {best_score:.4f} <= Threshold {self.threshold}")
            return True, results
            
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            return False, "System error during retrieval."
