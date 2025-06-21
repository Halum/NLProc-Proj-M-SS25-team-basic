"""
Specialized preprocessor module for handling extended document loading capabilities.

This module extends the baseline preprocessor functionality with additional
document loaders for specialized data formats like CSV files.
"""

from .document_reader import CSVDocumentLoader, SpecializedDocumentLoaderFactory

__all__ = [
    'CSVDocumentLoader',
    'SpecializedDocumentLoaderFactory'
]
