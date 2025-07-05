"""
Enhanced LLM module for specialized embeddings and text generation functionality.
This file provides OpenAI embeddings and chat model support, as well as Google Flan-T5 models
through HuggingFace integration for the specialization module.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import logging
import torch

from specialization.config.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL
)

logger = logging.getLogger(__name__)


class EnhancedLLM:
    """
    Enhanced LLM class for handling OpenAI embeddings and chat operations.
    This class supports OpenAI models and Google Flan-T5 models through HuggingFace.
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
        
        embedding_function = EnhancedLLM.embedding_function()
        embeddings = embedding_function.embed_documents(chunks)
        return embeddings
    
    @staticmethod
    def embedding_function():
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
        Get the chat model instance (OpenAI or HuggingFace).
        
        Args:
            model_name (str): Name of the chat model to use
            temperature (float): Temperature for text generation
            
        Returns:
            Union[ChatOpenAI, HuggingFacePipeline]: The chat model instance.
        """
        model_name = model_name or OPENAI_CHAT_MODEL
        
        # Check if it's a Google Flan-T5 model
        if model_name.startswith('google/flan-t5'):
            logger.info(f"Using HuggingFace Flan-T5 model: {model_name}")
            
            if torch.cuda.is_available():
                device = 0  # CUDA GPU id
                print("Using CUDA GPU")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using Apple MPS device")
            else:
                device = -1  # CPU
                print("Using CPU")
                
            
            # Create a text-generation pipeline for Flan-T5
            text_generation_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=temperature > 0.0,
                device=device
            )
            
            # For MPS, manually move model to mps device (optional)
            if isinstance(device, torch.device) and device.type == "mps":
                text_generation_pipeline.model.to(device)
            
            # Wrap it in LangChain's HuggingFacePipeline
            return HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        # Default to OpenAI for other models
        else:
            if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
                raise ValueError("Valid OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")

            logger.info(f"Using OpenAI chat model: {model_name}")
            return ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name=model_name,
                temperature=temperature
            )
