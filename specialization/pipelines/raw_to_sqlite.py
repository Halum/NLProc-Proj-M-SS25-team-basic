"""
SQLite pipeline for specialized document processing.

This pipeline stores documents in SQLite database with:
- Full-text search capabilities
- Metadata indexing
- Structured querying
- Hybrid retrieval support
"""

import os
import sys
import logging
import sqlite3
from typing import List, Dict, Any
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from specialization.config.config import SQLITE_OUTPUT_PATH, SQLITE_DB_NAME

class SQLitePipeline:
    """
    Pipeline for storing documents in SQLite database.
    """
    
    def __init__(self, db_name: str = SQLITE_DB_NAME):
        """
        Initialize the SQLite pipeline.
        
        Args:
            db_name: Name of the SQLite database file
        """
        self.db_name = db_name
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Run the SQLite pipeline.
        
        Args:
            input_dir: Directory containing input documents
            output_dir: Directory to save SQLite database
            
        Returns:
            Dictionary containing pipeline results and metrics
        """
        self.logger.info(f"Starting SQLite pipeline")
        self.logger.info(f"Input: {input_dir}")
        self.logger.info(f"Output: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "total_documents": 0,
            "total_chunks": 0,
            "database_file": self.db_name,
            "output_files": []
        }
        
        try:
            # Initialize database
            db_path = self._initialize_database(output_dir)
            results["output_files"].append(db_path)
            
            # Process and store documents
            stored_data = self._process_documents(input_dir, db_path)
            results.update(stored_data)
            
            self.logger.info(f"Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _initialize_database(self, output_dir: str) -> str:
        """Initialize SQLite database with schema."""
        self.logger.info("Initializing database...")
        db_path = os.path.join(output_dir, self.db_name)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Create full-text search index
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_search 
                USING fts5(content, metadata)
            """)
            
            conn.commit()
        
        return db_path
    
    def _process_documents(self, input_dir: str, db_path: str) -> Dict[str, Any]:
        """Process and store documents in database."""
        self.logger.info("Processing documents...")
        # TODO: Implement document processing and storage
        return {
            "total_documents": 0,
            "total_chunks": 0
        }