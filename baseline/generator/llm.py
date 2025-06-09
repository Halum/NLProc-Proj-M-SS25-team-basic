"""
Module for LLM and embedding functionality.
This file contains the LLM class that provides:
1. Text embedding generation using HuggingFace models
2. Language model invocation for generating text responses
3. Utility methods for handling embedding dimensions and model instantiation
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from baseline.config.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
)

class LLM:
    """
    Class for handling language model operations including embeddings generation,
    prompt creation, and text generation for question answering.
    """
    
    @staticmethod
    def generate_embedding(chunks):
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks (list): List of text chunks to embed.
            
        Returns:
            list: List of embeddings for the provided chunks.
        """
        embedding_model = LLM.embedding_model()
        embeddings = embedding_model.embed_documents(chunks)
        
        return embeddings
    
    @staticmethod
    def embedding_dimensions():
        """
        Get the dimensions of the embedding vectors.
        
        Returns:
            int: The dimensionality of the embedding vectors.
        """
        embedding_model = LLM.embedding_model()
        
        return embedding_model.client.get_sentence_embedding_dimension()
    
    @staticmethod
    def embedding_model():
        """
        Get the embedding model instance.
        
        Returns:
            HuggingFaceEmbeddings: The embedding model instance.
        """
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        return embedding_model
    
    @staticmethod
    def invoke_llm(prompt, creativity=False):
        """
        Invoke the language model to generate a response to the given prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            str: The generated answer from the LLM.
        """
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, low_cpu_mem_usage=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        
        num_beams = 1
        do_sample = False
        
        if creativity:
            num_beams = 2
            do_sample = True
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, num_beams=num_beams, do_sample=do_sample)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer