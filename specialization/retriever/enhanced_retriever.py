"""
Enhanced retriever module for specialized document processing.
This file contains the EnhancedRetriever class that extends the baseline Retriever with:
1. ChromaDB-based vector storage with metadata support
2. Enhanced document processing for structured data (movies)
3. Advanced search capabilities with filtering
4. Compatibility with baseline interface while adding new features
5. LangChain compatibility for RAG chains
"""

from re import L
import sys
import os
from typing import List, Dict, Any, Optional
import logging
import re

# Add baseline to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from specialization.generator.enhanced_llm import EnhancedLLM
from specialization.retriever.vector_store_chroma import VectorStoreChroma
from specialization.config.config import VECTOR_COLLECTION_NAME, VECTOR_PERSIST_DIRECTORY, METADATA_COLUMNS, CHUNKING_COLUMN


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def _prepare_document_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare document metadata for storage and filtering.
        
        Args:
            metadata (Dict[str, Any]): Original document metadata
            
        Returns:
            Dict[str, Any]: Enhanced metadata with additional fields for filtering
        """
        # Create a copy to avoid modifying the original
        prepared = metadata.copy()
        
        # Process release dates to add timestamps
        if 'release_date' in prepared and prepared['release_date']:
            try:
                # Keep the original date string
                date_str = str(prepared['release_date']).strip()
                
                # Skip empty or invalid values
                if date_str and date_str.lower() not in ['nan', 'none', 'null', '']:
                    # Extract year, month, day manually - more reliable than dateutil
                    date_match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
                    
                    if date_match:
                        # Get year, month, day from regex match
                        year = int(date_match.group(1))
                        month = int(date_match.group(2))
                        day = int(date_match.group(3))
                        
                        # Simple validation to avoid invalid dates
                        if 1800 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                            # Store as numeric year for filtering (more reliable than timestamp)
                            prepared['release_year'] = year
                            
                            # Calculate an integer value for sorting (YYYYMMDD format)
                            prepared['release_date_sortable'] = year * 10000 + month * 100 + day
                    else:
                        # Fall back to year extraction for non-standard formats
                        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
                        if year_match:
                            year = int(year_match.group(1))
                            prepared['release_year'] = year
                            prepared['release_date_sortable'] = year * 10000  # Just the year
            except Exception as e:
                print(f"Error processing release date '{prepared['release_date']}': {type(e).__name__} - {e}")
        
        # Process numeric fields to ensure they're stored as numbers
        if 'revenue' in prepared and prepared['revenue']:
            try:
                if isinstance(prepared['revenue'], str) and prepared['revenue'].lower() in ['nan', 'none', 'null', '']:
                    prepared['revenue'] = 0
                else:
                    prepared['revenue'] = float(prepared['revenue'])
            except (ValueError, TypeError):
                prepared['revenue'] = 0
                
        if 'runtime' in prepared and prepared['runtime']:
            try:
                if isinstance(prepared['runtime'], str) and prepared['runtime'].lower() in ['nan', 'none', 'null', '']:
                    prepared['runtime'] = 0
                else:
                    prepared['runtime'] = int(float(prepared['runtime']))
            except (ValueError, TypeError):
                prepared['runtime'] = 0
        
        return prepared
    
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
            
            # Prepare metadata - add this function to process dates
            doc_metadata = self._prepare_document_metadata(doc_metadata)
        
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
    
    def add_documents(self, processed_data: List[Dict[str, Any]]):
        """
        Add processed documents with metadata (specialized functionality).
        
        Args:
            processed_data (List[Dict[str, Any]]): List of processed document dictionaries.
            text_column (str): Name of the column containing text to be chunked.
        """
        documents = []
        metadatas = []
        
        for item in processed_data:
            if CHUNKING_COLUMN in item and item[CHUNKING_COLUMN]:
                documents.append(item[CHUNKING_COLUMN])
                # Create metadata from all other columns
                metadata = {}
                for k, v in item.items():
                    if k in METADATA_COLUMNS:
                        # Lowercase string values for efficient matching
                        if isinstance(v, str):
                            metadata[k] = v.lower()
                        # Convert lists to strings for ChromaDB compatibility
                        elif isinstance(v, list):
                            metadata[k] = ', '.join(str(v) for v in v)
                        else:
                            metadata[k] = v

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
        # Preprocess filter values to match metadata preprocessing
        processed_filter = None
        if filter_dict:
            processed_filter = {}
            for key, value in filter_dict.items():
                if isinstance(value, str):
                    processed_filter[key] = value.lower()
                else:
                    processed_filter[key] = value
        
        return self.__vector_store.search(query, k=k, filter_dict=processed_filter)
    
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
