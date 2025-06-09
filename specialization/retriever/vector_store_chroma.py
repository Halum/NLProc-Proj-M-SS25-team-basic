"""
Module for ChromaDB vector database functionality.
This file contains the VectorStoreChroma class that provides:
1. ChromaDB-based vector storage and retrieval operations
2. Efficient similarity search for finding relevant documents with metadata
3. Persistence capabilities with automatic saving
4. Enhanced metadata support for specialized retrieval pipeline
"""

import os
import uuid
import shutil
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreChroma:
    """
    Vector store implementation using ChromaDB.
    Provides methods for storing and retrieving vector embeddings with metadata support.
    
    This class maintains API compatibility with the baseline VectorStoreFaiss while
    adding enhanced features like metadata storage and filtering.
    """
    
    def __init__(self, embedding_dim: int, collection_name: str = "default_collection", 
                 persist_directory: str = "chroma_db", embedding_model: str = "all-mpnet-base-v2"):
        """
        Initialize the ChromaDB vector store with specified embedding dimensions.
        
        Args:
            embedding_dim (int): Dimensionality of the embedding vectors.
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist the database.
            embedding_model (str): Name of the embedding model to use.
        """
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize LangChain Chroma vector store (handles persistence automatically)
        self.__db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory
        )


    def cleanup(self):
        """
        Reset the vector store, removing all stored embeddings and cleaning up directories.
        """
        try:
            # Delete the collection through LangChain Chroma
            self.__db.delete_collection()
            
            # Clean up the entire persist directory to remove orphaned UUID directories
            if os.path.exists(self.persist_directory):
                # Remove all contents of the persist directory
                for item in os.listdir(self.persist_directory):
                    item_path = os.path.join(self.persist_directory, item)
                    if os.path.isdir(item_path):
                        # Remove UUID directories that ChromaDB creates for collections
                        import shutil
                        shutil.rmtree(item_path)
                    elif item != 'chroma.sqlite3':
                        # Keep the main database file but remove other files if needed
                        os.remove(item_path)
            
            # Recreate the collection with a fresh start
            self.__db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
            # If cleanup fails, still try to recreate the collection
            try:
                self.__db = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
            except Exception as e2:
                print(f"Error recreating collection: {e2}")
    
    def search(self, query: str, k: int = 5, 
                           filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Enhanced search method that returns results with metadata.
        
        Args:
            query (str): Query text to search for.
            k (int): Number of results to return.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter dictionary.
            
        Returns:
            List[Dict[str, Any]]: List of search results with content and metadata.
        """
        # Perform similarity search with metadata
        results = self.__db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dict[str, Any]: Collection information.
        """
        try:
            # Access collection info through LangChain Chroma
            collection = self.__db._collection
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata
            }
        except Exception as e:
            return {'error': str(e)}
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, 
                     ids: List[str] = None):
        """
        Enhanced method to add documents with texts and metadata directly.
        Supports batch processing to handle large datasets.
        
        Args:
            texts (List[str]): List of text documents to add.
            metadatas (List[Dict[str, Any]], optional): List of metadata dictionaries.
            ids (List[str], optional): List of document IDs.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Import batch size from config to avoid circular imports
        try:
            from specialization.config.config import EMBEDDING_BATCH_SIZE
            batch_size = EMBEDDING_BATCH_SIZE
        except ImportError:
            batch_size = 5000  # Default fallback
        
        # Process in batches to avoid ChromaDB limits
        total_texts = len(texts)
        for i in range(0, total_texts, batch_size):
            end_idx = min(i + batch_size, total_texts)
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            # Add this batch to the vector store
            self.__db.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            print(f"Processed batch {i//batch_size + 1}: {len(batch_texts)} documents (Total: {end_idx}/{total_texts})")
