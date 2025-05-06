import os
from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load a document from the given path."""
        pass