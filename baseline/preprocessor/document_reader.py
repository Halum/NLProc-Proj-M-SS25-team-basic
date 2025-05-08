import os
from abc import ABC, abstractmethod
import PyPDF2
from pathlib import Path
import docx


class DocumentLoaderAbs(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load a document from the given path."""
        pass


class PDFDocumentLoader(DocumentLoaderAbs):
    """Loader for PDF documents."""

    def load(self, file_path: str) -> str:
        """Load a PDF document."""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text


class TextDocumentLoader(DocumentLoaderAbs):
    """Loader for text documents."""

    def load(self, file_path: str) -> str:
        """Load a text document."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


class DocxDocumentLoader(DocumentLoaderAbs):
    """Loader for DOCX documents."""

    def load(self, file_path: str) -> str:
        """Load a DOCX or Word document."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text


class DocumentLoaderFactory:
    """
    Factory class to create appropriate DocumentLoader instances based on file type.
    """

    @staticmethod
    def get_loader(file_path: str) -> DocumentLoaderAbs:
        """
        Get the appropriate loader for the given file type.

        Args:
            file_path (str): Path to the document file.

        Returns:
            DocumentLoaderAbs: An instance of a subclass of DocumentLoaderAbs.
        """
        extension = Path(file_path).suffix.lower()
        if extension == ".txt" or extension == ".md":
            return TextDocumentLoader()
        elif extension == ".pdf":
            return PDFDocumentLoader()
        elif extension == ".docx":
            return DocxDocumentLoader()
        else:
            raise ValueError(f"Unsupported file type: {extension}")


class DocumentReader:
    @staticmethod
    def read_document(file_path: str) -> str:
        """
        Load the content of a document using the appropriate loader.

        Args:
            file_path (str): Path to the document file.

        Returns:
            str: Content of the document as a string.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        loader = DocumentLoaderFactory.get_loader(file_path)
        
        document = loader.load(file_path)
        
        return document


    @staticmethod
    def read_documents_in_dir(directory: str) -> list:
        documents = []    
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                try:
                    file_content = DocumentReader.read_document(file_path)
                    documents.append(file_content)
                except Exception as e:
                    print(f"[ERROR] Failed to process '{filename}': {e}")
        
        return documents
    