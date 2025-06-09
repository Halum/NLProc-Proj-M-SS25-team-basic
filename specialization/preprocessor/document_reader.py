#!/usr/bin/env python3
"""
Extended Document Reader for Specialization Track

This module extends the baseline document reader with additional loaders
for specialized formats like CSV files while maintaining compatibility
with the baseline DocumentLoaderFactory.

Features:
- CSV document loader with pandas DataFrame support
- Extended factory that falls back to baseline for other formats
- Robust error handling and logging
"""

import pandas as pd
from pathlib import Path
import logging

from baseline.preprocessor.document_reader import DocumentLoaderAbs, DocumentLoaderFactory

logger = logging.getLogger(__name__)


class CSVDocumentLoader(DocumentLoaderAbs):
    """Extended document loader for CSV files in specialization directory."""
    
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and return as pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded CSV data
            
        Raises:
            Exception: If CSV file cannot be loaded
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded CSV: {file_path} with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise


class SpecializedDocumentLoaderFactory(DocumentLoaderFactory):
    """
    Extended factory class that includes CSV support while maintaining
    compatibility with baseline document loaders.
    
    This factory extends the baseline DocumentLoaderFactory to handle
    additional file formats specific to the specialization track.
    """
    
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoaderAbs:
        """
        Get the appropriate loader for the given file type, including CSV support.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            DocumentLoaderAbs: An instance of a document loader
            
        Raises:
            ValueError: If file extension is not supported
        """
        extension = Path(file_path).suffix.lower()
        
        if extension == ".csv":
            return CSVDocumentLoader()
        else:
            # Fall back to baseline factory for other file types
            return DocumentLoaderFactory.get_loader(file_path)
