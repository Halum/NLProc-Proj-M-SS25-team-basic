from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    """

    @abstractmethod
    def chunk(self, text: str) -> list:
        """
        Chunk the given text into smaller pieces.
        """
        pass
    