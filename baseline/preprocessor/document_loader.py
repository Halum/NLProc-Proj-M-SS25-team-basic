import os
from abc import ABC, abstractmethod
import PyPDF2
from pathlib import Path

class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load a document from the given path."""
        pass

class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF documents."""

    def load(self, file_path: str) -> str:
        """Load a PDF document."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    
class TextDocumentLoader(DocumentLoader):
    """Loader for text documents."""

    def load(self, file_path: str) -> str:
        """Load a text document."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class DocumentLoaderFactory:
    """
    Factory class to create appropriate DocumentLoader instances based on file type.
    """

    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """
        Get the appropriate loader for the given file type.

        Args:
            file_path (str): Path to the document file.

        Returns:
            DocumentLoader: An instance of a subclass of DocumentLoader.
        """
        extension = Path(file_path).suffix.lower()
        if extension == '.txt':
            return TextDocumentLoader()
        elif extension == '.pdf':
            return PDFDocumentLoader()

        else:
            raise ValueError(f"Unsupported file type: {extension}")
