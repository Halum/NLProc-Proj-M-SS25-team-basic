import faiss
import numpy as np

class VectorStoreFaiss:
    """
    Vector store implementation using Facebook AI Similarity Search (FAISS).
    Provides methods for storing and retrieving vector embeddings.
    """
    
    def __init__(self, embedding_dim):
        """
        Initialize the vector store with specified embedding dimensions.
        
        Args:
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        self.__db = faiss.IndexFlatL2(embedding_dim)
        
    def add(self, embeddings):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings (list): List of embedding vectors to add to the store.
        """
        self.__db.add(np.array(embeddings))
        
    def cleanup(self):
        """
        Reset the vector store, removing all stored embeddings.
        """
        self.__db.reset()
        
    def search(self, query_embedding, k=5):
        """
        Search for similar vectors in the vector store.
        
        Args:
            query_embedding (list): The query embedding vector.
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.
            
        Returns:
            tuple: (distances, indices) where distances are the L2 distances to the
                  nearest neighbors and indices are their positions in the database.
        """
        distances, indices = self.__db.search(np.array(query_embedding), k=k)
        
        return distances, indices
    
    def save_index(self, index_path):
        """
        Save the current state of the vector store to a file.
        
        Args:
            index_path (str): Path to save the index file.
        """
        faiss.write_index(self.__db, f"{index_path}.faiss")
        
    def load_index(self, index_path):
        """
        Load a vector store index from a file.
        
        Args:
            index_path (str): Path to the index file to load.
        """
        
        self.__db = faiss.read_index(f"{index_path}.faiss")