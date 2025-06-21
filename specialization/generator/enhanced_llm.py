"""
Enhanced LLM module for specialized embeddings and text generation functionality.
This file provides OpenAI embeddings and chat model support for the specialization module.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import logging

from specialization.config.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL
)

logger = logging.getLogger(__name__)


class EnhancedLLM:
    """
    Enhanced LLM class for handling OpenAI embeddings and chat operations.
    This class is specialized for OpenAI models only.
    """
    
    @staticmethod
    def generate_embedding(chunks):
        """
        Generate embeddings for a list of text chunks using OpenAI.
        
        Args:
            chunks (list): List of text chunks to embed.
            
        Returns:
            list: List of embeddings for the provided chunks.
        """
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        embedding_model = EnhancedLLM.embedding_model()
        embeddings = embedding_model.embed_documents(chunks)
        return embeddings
    
    @staticmethod
    def embedding_dimensions():
        """
        Get the dimensions of the OpenAI embedding vectors.
        
        Returns:
            int: The dimensionality of the embedding vectors (1536 for text-embedding-ada-002).
        """
        # OpenAI text-embedding-ada-002 has 1536 dimensions
        return 1536
    
    @staticmethod
    def embedding_model():
        """
        Get the OpenAI embedding model instance.
        
        Returns:
            OpenAIEmbeddings: The OpenAI embedding model instance.
        """
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
            raise ValueError("Valid OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        logger.info(f"Using OpenAI embeddings model: {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=OPENAI_EMBEDDING_MODEL
        )
    
    @staticmethod
    def chat_model(model_name: str = None, temperature: float = 0.0):
        """
        Get the OpenAI chat model instance.
        
        Args:
            model_name (str): Name of the OpenAI chat model to use
            temperature (float): Temperature for text generation
            
        Returns:
            ChatOpenAI: The OpenAI chat model instance.
        """
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
            raise ValueError("Valid OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")

        model_name = model_name or OPENAI_CHAT_MODEL

        logger.info(f"Using OpenAI chat model: {model_name}")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=temperature
        )
