"""
Module for text chunking strategies.
This file contains a collection of classes that provide:
1. Different document chunking strategies using various approaches
2. An extensible framework for implementing custom chunking algorithms
3. Preprocessing techniques for improving text segmentation quality
4. Integration with NLP libraries for semantic and structural chunking
"""

from abc import ABC, abstractmethod
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
)
from baseline.generator.llm import LLM
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Tuple
import nltk

nltk.download("punkt", quiet=True)


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
    Chunking strategy that splits text into fixed-size chunks using LangChain.
    """

    def __init__(self, chunk_size: int):
        """
        Initialize the fixed-size chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separator=""
        )

    def chunk(self, text: str) -> list:
        """
        Chunk the text into fixed-size pieces using LangChain.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        return self.splitter.split_text(text)


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
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap
        ).split_text(text)

        return chunks


class SentenceBasedChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text into sentences using LangChain NLTK splitter.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the sentence chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size
        self.splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0
        )

    def chunk(self, text: str) -> list:
        """
        Chunk the text into sentences using LangChain NLTK splitter.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        return self.splitter.split_text(text)


class ParagraphBasedChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text into paragraphs using LangChain.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the paragraph-based chunking strategy.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separator="\n\n"
        )

    def chunk(self, text: str) -> list:
        """
        Chunk the text into paragraphs using LangChain.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        return self.splitter.split_text(text)


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that uses a recursive approach to split text into smaller pieces.
    """

    def chunk(self, text: str) -> list:
        """
        Chunk the text using a recursive approach.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        embedding_model = LLM.embedding_model()
        splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type='percentile'
        )
        
        chunks = splitter.split_text(text)    

        return chunks


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
