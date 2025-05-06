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
    
class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text into fixed-size chunks.
    """

    def __init__(self, chunk_size: int):
        """
        Initialize the fixed-size chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list:
        """
        Chunk the text into fixed-size pieces.
        
        Args:
            text (str): The text to be chunked.
            
        Returns:
            list: A list of text chunks.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]