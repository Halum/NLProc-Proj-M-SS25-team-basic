"""
Module for implementing answer generation components.
This file contains the Generator class that provides functionality for:
1. Building effective prompts for LLM-based answer generation
2. Generating answers to user queries based on relevant retrieved context
3. Interfacing with the LLM service to produce contextually accurate responses
"""

from generator.llm import LLM

class Generator:
    @staticmethod
    def build_answering_prompt(query, relevant_contexts):
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
    def generate_answer(answering_prompt):
        """
        Generate an answer using the LLM based on the provided prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            str: The generated answer from the LLM.
        """
        answer = LLM.invoke_llm(answering_prompt)
        
        return answer