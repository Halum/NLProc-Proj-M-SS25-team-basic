import dis
import faiss
import numpy as np

class VectorStoreFaiss:
    def __init__(self, embedding_dim):
        self.__db = faiss.IndexFlatL2(embedding_dim)
        
    def add(self, embeddings):
        self.__db.add(np.array(embeddings))
        
    def cleanup(self):
        self.__db.reset()
        
    def search(self, query_embedding, k=5):
        distances, indices = self.__db.search(np.array(query_embedding), k=k)
        
        return distances, indices