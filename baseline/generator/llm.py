from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config.config import (
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
    def generate_answering_prompt(query, relevant_contexts):
        """
        Generate a prompt for the LLM to answer a question using relevant contexts.
        
        Args:
            query (str): The question to be answered.
            relevant_contexts (list): List of relevant text chunks to use as context.
            
        Returns:
            str: A formatted prompt for the LLM.
        """
        context  = "\n\n".join(relevant_contexts)
        
        prompt = f"""You are a helpful assistant. Use the following context to provide to the point answer of the question.

        Context:
        {context}

        Question:
        {query}

        Answer:"""
        
        return prompt
    
    @staticmethod
    def invoke_llm(prompt):
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
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
            
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer