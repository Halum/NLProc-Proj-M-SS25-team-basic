"""
Module for document loading and parsing functionality using LangChain.
Provides:
1. Abstraction for loading different document formats (PDF, TXT, DOCX, CSV)
2. Unified interface for extracting text from various file types
3. Directory traversal for batch document processing
"""

import os
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, PDFPlumberLoader, Docx2txtLoader, CSVLoader
)

SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'csv']

class DocumentLoaderFactory:
    """
    Factory class to create appropriate LangChain DocumentLoader instances based on file type.
    """

    @staticmethod
    def get_loader(file_path: str):
        extension = Path(file_path).suffix.lower()
        if extension == ".txt" or extension == ".md":
            # Use plain Python open for .txt files for better error handling
            return None
        elif extension == ".pdf":
            return PDFPlumberLoader(file_path)
        elif extension == ".docx":
            return Docx2txtLoader(file_path)
        elif extension == ".csv":
            return CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

class DocumentReader:
    @staticmethod
    def read_document(file_path: str) -> str:
        """
        Load the content of a document using the appropriate LangChain loader.

        Args:
            file_path (str): Path to the document file.

        Returns:
            str: Content of the document as a string.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        extension = Path(file_path).suffix.lower()
        if extension == ".txt" or extension == ".md":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                raise RuntimeError(f"Error reading text file '{file_path}': {e}")
        else:
            loader = DocumentLoaderFactory.get_loader(file_path)
            docs = loader.load()
            # Concatenate all document contents for compatibility
            return "\n".join(doc.page_content for doc in docs)

    @staticmethod
    def read_documents_in_dir(directory: str) -> list:
        """
        Read all documents in the specified directory.

        Args:
            directory (str): Path to the directory containing documents.

        Returns:
            list: List of document content strings from all successfully processed files.
        """
        documents = []

        for filename in os.listdir(directory):
            file_type = filename.split('.')[-1].lower()

            if file_type not in SUPPORTED_FILE_TYPES:
                print(f"[WARNING] Unsupported file type '{file_type}' for file '{filename}'. Skipping.")
                continue

            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                try:
                    file_content = DocumentReader.read_document(file_path)
                    documents.append(file_content)
                except Exception as e:
                    print(f"[ERROR] Failed to process '{filename}': {e}")
                    import traceback
                    traceback.print_exc()

        return documents