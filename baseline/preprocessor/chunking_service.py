"""
Module for text chunking strategies using LangChain.
Provides:
1. Different document chunking strategies using various approaches
2. An extensible framework for implementing custom chunking algorithms
3. Preprocessing techniques for improving text segmentation quality
4. Integration with NLP libraries for semantic and structural chunking
"""

import nltk
from abc import ABC, abstractmethod
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from typing import List, Tuple

# Download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list:
        pass

class FixedSizeChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    def chunk(self, text: str) -> list:
        return self.splitter.split_text(text)

class SlidingWindowChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int, overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def chunk(self, text: str) -> list:
        return self.splitter.split_text(text)

class SentenceBasedChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list:
        sentences = nltk.sent_tokenize(text)
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
    def __init__(self, chunk_size: int = 1000):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, separators=["\n\n"]
        )

    def chunk(self, text: str) -> list:
        return self.splitter.split_text(text)

class MarkdownHeaderChunkingStrategy(ChunkingStrategy):
    def __init__(self, headers_to_split_on: List[Tuple[str, str]] = None):
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3"),
                ("####", "Header4"),
                ("#####", "Header5"),
                ("######", "Header6"),
            ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    def chunk(self, text: str) -> List[str]:
        documents = self.splitter.split_text(text)
        return [doc.page_content for doc in documents]