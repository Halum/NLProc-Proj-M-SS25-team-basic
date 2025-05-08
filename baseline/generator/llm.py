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
    
    @staticmethod
    def embedding_dimensions():
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        return embedding_model.get_sentence_embedding_dimension()