"""
Module for vector database functionality using LangChain.
This file contains the VectorStoreFaiss class that provides:
1. FAISS-based vector storage and retrieval operations via LangChain
2. Efficient similarity search for finding relevant documents
3. Persistence capabilities to save and load vector indices
4. Low-level vector operations for the retrieval pipeline
"""

from langchain_community.vectorstores import FAISS
import numpy as np
import os

class VectorStoreFaiss:
    """
    Vector store implementation using LangChain's FAISS wrapper.
    Provides methods for storing and retrieving vector embeddings.
    """

    def __init__(self, embedding_model):
        """
        Initialize the vector store with a LangChain embedding model.

        Args:
            embedding_model: LangChain embedding model instance.
        """
        self.embedding_model = embedding_model
        self.db = None
        self.texts = []

    def add(self, texts):
        """
        Add texts to the vector store (embeddings are computed internally).

        Args:
            texts (list): List of text chunks to add to the store.
        """
        if not texts:
            return
        if self.db is None:
            self.db = FAISS.from_texts(texts, self.embedding_model)
            self.texts.extend(texts)
        else:
            new_db = FAISS.from_texts(texts, self.embedding_model)
            self.db.merge_from(new_db)
            self.texts.extend(texts)

    def cleanup(self):
        """
        Reset the vector store, removing all stored embeddings.
        """
        self.db = None
        self.texts = []

    def search(self, query, k=5):
        """
        Search for similar vectors in the vector store.

        Args:
            query (str): The query text.
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.

        Returns:
            list: List of (text, score) tuples for the top-k matches.
        """
        if self.db is None:
            return []
        results = self.db.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]

    def save_index(self, index_path):
        """
        Save the current state of the vector store to a file.

        Args:
            index_path (str): Path to save the index file.
        """
        if self.db is not None:
            self.db.save_local(index_path)

    def load_index(self, index_path):
        """
        Load a vector store index from a file.

        Args:
            index_path (str): Path to the index file to load.
        """
        self.db = FAISS.load_local(index_path, self.embedding_model)