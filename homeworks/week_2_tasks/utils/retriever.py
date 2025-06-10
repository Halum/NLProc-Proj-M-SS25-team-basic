import faiss
import numpy as np

from text_embedding.generate_embedding import generate_embedding
from text_embedding.load_model import load_transformer_model

from config import (
    EMBEDDING_MODEL,
)

def store_embeddings(embeddings):
    db = faiss.IndexFlatL2(embeddings[0].shape[0])
    db.add(np.array(embeddings))
    
    return db

def retrieve_similar_embeddings(db, query, k=5):
    """Retrieve the top k most similar embeddings from the database."""
    query_embedding = generate_document_embeddings([query])
    distances, indices = db.search(np.array(query_embedding), k=k)
    return distances, indices

def get_retrieved_chunks(indices, file_contents):
    """Retrieve the chunks corresponding to the indices."""
    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append(file_contents[i])
    return retrieved_chunks

def generate_document_embeddings(file_contents):
    """Generate embeddings from file contents using the configured model."""
    model = load_transformer_model(EMBEDDING_MODEL)
    embeddings = generate_embedding(file_contents, model)
    
    return embeddings