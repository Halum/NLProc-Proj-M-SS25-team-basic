#!/usr/bin/env python3
"""
Alternative implementation using the Enhanced Retriever pattern.
This demonstrates how the pipeline can also be implemented using the 
retriever pattern that matches the baseline architecture.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.preprocessor.chunking_service import FixedSizeChunkingStrategy
from specialization.retriever.enhanced_retriever import EnhancedRetriever
from specialization.utils.data_utils import filter_json_columns, concatenate_columns_to_chunking
from specialization.config.config import (
    PROCESSED_DOCUMENT_DIR_PATH,
    PROCESSED_DOCUMENT_NAME,
    DATA_COLUMNS_TO_KEEP,
    CHUNK_SIZE,
    ADD_TO_CHUNKING_COLUMN,
    CHUNKING_COLUMN
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProcessedToEmbeddingsRetrieverPipeline:
    """
    Alternative implementation using the Enhanced Retriever pattern.
    This follows the baseline architecture more closely.
    """
    
    def __init__(self, processed_data_path: str = None, chunk_size: int = None):
        """
        Initialize the pipeline with configurable parameters.
        
        Args:
            processed_data_path (str): Path to processed data file
            chunk_size (int): Size for text chunking
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunking_strategy = FixedSizeChunkingStrategy(chunk_size=self.chunk_size)
        self.retriever = EnhancedRetriever(self.chunking_strategy, fresh_db=True)
        
        # Set processed data path
        if processed_data_path:
            self.processed_data_path = Path(processed_data_path)
        else:
            self.processed_data_path = Path(PROCESSED_DOCUMENT_DIR_PATH) / PROCESSED_DOCUMENT_NAME
            
        logger.info(f"Pipeline initialized - Processed data: {self.processed_data_path}")
        logger.info(f"Chunk size: {self.chunk_size}, Columns to keep: {DATA_COLUMNS_TO_KEEP}")
        
    def load_processed_data(self) -> List[Dict[str, Any]]:
        """
        Load processed JSON data and filter to keep only specified columns.
        
        Returns:
            List[Dict[str, Any]]: Filtered processed documents
        """
        logger.info(f"Loading processed data from {self.processed_data_path}")
        
        with open(self.processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter to keep only specified columns using utility function
        filtered_data = filter_json_columns(data, DATA_COLUMNS_TO_KEEP)
        
        # Concatenate additional columns to chunking column using ADD_TO_CHUNKING_COLUMN
        enhanced_data = concatenate_columns_to_chunking(filtered_data, CHUNKING_COLUMN, ADD_TO_CHUNKING_COLUMN)
        
        logger.info(f"Loaded {len(enhanced_data)} documents with required columns and enhanced chunking content")
        return enhanced_data
    
    def run(self) -> str:
        """
        Execute the complete pipeline using the retriever pattern.
        
        Returns:
            str: Information about the completed pipeline
        """
        logger.info("Starting enhanced embeddings pipeline")
        
        try:
            # Step 1: Load, filter, and enhance processed data
            documents = self.load_processed_data()
            
            # Step 2: Add processed documents to retriever
            chunks_count = self.retriever.add_documents(documents)
            logger.info(f"Generated {chunks_count} chunks")
            
            # Step 5: Get collection info
            return  self.retriever.get_collection_info()

        except Exception as e:
            raise e


def main():
    """Main function to run the pipeline."""
    try:
        pipeline = ProcessedToEmbeddingsRetrieverPipeline()
        results = pipeline.run()
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📊 Results: {results}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
