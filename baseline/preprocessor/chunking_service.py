from abc import ABC, abstractmethod
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt_tab", quiet=True)


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
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=0
        ).split_text(text)
        return splitter


class SlidingWindowChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that uses a sliding window approach.
    """

    def __init__(self, chunk_size: int, overlap: int):
        """
        Initialize the sliding window chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
            overlap (int): The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list:
        """
        Chunk the text using a sliding window approach.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap
        ).split_text(text)

        return splitter


class SentenceBasedChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text into sentences.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the sentence chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list:
        """
        Chunk the text into sentences.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class ParagraphBasedChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text into paragraphs.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the paragraph-based chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list:
        """
        Chunk the text into paragraphs.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that uses a recursive approach to split text into smaller pieces.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the recursive chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list:
        """
        Chunk the text using a recursive approach.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        ).split_text(text)

        return splitter


class MarkdownHeaderChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits Markdown text based on header structure.
    """
    def __init__(self, headers_to_split_on: List[Tuple[str, str]] = None):
        """
        Initialize the markdown header chunking strategy.

        Args:
            headers_to_split_on (List[Tuple[str, str]]): A list of tuples specifying
                the markdown headers to split on and their corresponding metadata keys.
                Example: [("#", "Header1"), ("##", "Header2")]
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3"),
                ("####", "Header4"),
                ("#####", "Header5"),
                ("######", "Header6"),
            ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

    def chunk(self, text: str) -> List[str]:
        """
        Chunk the markdown text into sections based on headers.

        Args:
            text (str): The markdown text to be chunked.

        Returns:
            List[str]: A list of markdown chunks as strings.
        """
        documents = self.splitter.split_text(text)
        chunks = [doc.page_content for doc in documents]
        return chunks
