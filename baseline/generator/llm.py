from sentence_transformers import SentenceTransformer

from config.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
)

class LLM:
    @staticmethod
    def generate_embedding(chunks):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embedding_model.encode(chunks)
        
        return embeddings