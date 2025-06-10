"""
Enhanced retriever module for specialized document processing.
This file contains the EnhancedRetriever class that extends the baseline Retriever with:
1. ChromaDB-based vector storage with metadata support
2. Enhanced document processing for structured data (movies)
3. Advanced search capabilities with filtering
4. Compatibility with baseline interface while adding new features
5. LangChain compatibility for RAG chains
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add baseline to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from specialization.generator.enhanced_llm import EnhancedLLM
from specialization.retriever.vector_store_chroma import VectorStoreChroma
from specialization.config.config import VECTOR_COLLECTION_NAME, VECTOR_PERSIST_DIRECTORY


class EnhancedRetriever:
    """
    Enhanced class for document retrieval using ChromaDB vector embeddings.
    Extends baseline functionality while maintaining API compatibility.
    Handles document processing, chunking, embedding, storage, and querying with metadata support.
    """
    
    def __init__(self, chunking_strategy, fresh_db: bool = False):
        """
        Initialize the EnhancedRetriever with a specified chunking strategy.
        
        Args:
            chunking_strategy: Strategy object for dividing documents into chunks.
            fresh_db (bool): Flag indicating whether to use a fresh database.
        """
        self.chunking_strategy = chunking_strategy

        # Initialize ChromaDB vector store with proper dimensions
        embedding_dim = self._get_embedding_dimensions()
        self.__vector_store = VectorStoreChroma(
            embedding_dim=embedding_dim,
            collection_name=VECTOR_COLLECTION_NAME,
            persist_directory=VECTOR_PERSIST_DIRECTORY
        )
        
        if fresh_db:
            # If fresh_db is True, reset the vector store
            self.__reset__()
        
    def _get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions from the EnhancedLLM module."""
        try:
            return EnhancedLLM.embedding_dimensions()
        except (AttributeError, ImportError, Exception) as e:
            raise e
        
    def __reset__(self):
        """
        Reset the retriever by clearing the vector store.
        """
        self.__vector_store.cleanup()
    
    def _process_and_store_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Process documents by chunking them and storing in ChromaDB.
        
        Args:
            documents (List[str]): List of document texts.
            metadatas (List[Dict[str, Any]], optional): List of metadata for each document.
            
        Returns:
            list: List of document contents.
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
            
        chunk_texts = []
        chunk_metadatas = []
        
        for i, document in enumerate(documents):
            # Get document metadata
            doc_metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Generate chunks for this document
            chunks = self.chunking_strategy.chunk(document)
            
            for j, chunk in enumerate(chunks):
                chunk_texts.append(chunk)
                # Create metadata for this chunk (inherit document metadata)
                chunk_meta = {**doc_metadata}
                chunk_metadatas.append(chunk_meta)
        
        # Store chunks in ChromaDB
        self.__vector_store.add_documents(
            texts=chunk_texts,
            metadatas=chunk_metadatas
        )
    
    def add_documents(self, processed_data: List[Dict[str, Any]], chunk_column: str):
        """
        Add processed documents with metadata (specialized functionality).
        
        Args:
            processed_data (List[Dict[str, Any]]): List of processed document dictionaries.
            text_column (str): Name of the column containing text to be chunked.
        """
        documents = []
        metadatas = []
        
        for item in processed_data:
            if chunk_column in item and item[chunk_column]:
                documents.append(item[chunk_column])
                # Create metadata from all other columns
                metadata = {k: v for k, v in item.items() if k != chunk_column}
                # Convert lists to strings for ChromaDB compatibility
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ', '.join(str(v) for v in value)
                metadatas.append(metadata)
        
        self._process_and_store_documents(documents, metadatas)
        
        # Get collection info to verify preprocessing worked
        db_info = self.get_collection_info()
        
        return db_info.get('count', 0)

    def query(self, query: str, k: int = 5, 
                           filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Enhanced query method that returns results with metadata.
        
        Args:
            query (str): The query to search for.
            k (int): Number of results to return.
            filter_dict (Dict[str, Any]): Optional metadata filter.
            
        Returns:
            List[Dict[str, Any]]: List of results with content, metadata, and scores.
        """
        return self.__vector_store.search(query, k=k, filter_dict=filter_dict)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dict[str, Any]: Collection information.
        """
        return self.__vector_store.get_collection_info()
    
    def cleanup(self):
        """
        Clean up the vector store and reset all data.
        """
        self.__reset__()
    
    def as_langchain_retriever(self, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
        """
        Create a LangChain-compatible retriever wrapper.
        
        Args:
            k (int): Number of documents to retrieve
            filter_dict (Optional[Dict[str, Any]]): Metadata filter
            
        Returns:
            LangChainRetrieverWrapper: LangChain-compatible retriever
        """
        return LangChainRetrieverWrapper(self, k=k, filter_dict=filter_dict)


class LangChainRetrieverWrapper(BaseRetriever):
    """
    LangChain-compatible wrapper for the EnhancedRetriever.
    """
    
    def __init__(self, enhanced_retriever: EnhancedRetriever, k: int = 5, 
                 filter_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.enhanced_retriever = enhanced_retriever
        self.k = k
        self.filter_dict = filter_dict
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents using the EnhancedRetriever.
        
        Args:
            query (str): The query string
            run_manager: LangChain callback manager
            
        Returns:
            List[Document]: List of LangChain Document objects
        """
        results = self.enhanced_retriever.query(
            query=query, 
            k=self.k, 
            filter_dict=self.filter_dict
        )
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result['content'],
                metadata={
                    **result['metadata'],
                    'score': result['score']
                }
            )
            documents.append(doc)
        
        return documents
