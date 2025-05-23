"""
Module for answer verification and evaluation.
This file contains the AnswerVerifier class that provides:
1. Methods for comparing generated answers against ground truth
2. Utilities for loading labeled test data from external sources
"""

from config.config import TEST_QUESTIONS_PATH
import json

class AnswerVerifier:
    """
    Class for verifying and evaluating the correctness of generated answers
    by comparing them with labeled data and context.
    """
    
    @staticmethod
    def get_sample_labeled_data():
        """
        Retrieve sample labeled data for testing and evaluation purposes.
        
        Returns:
            dict: A dictionary containing query, expected answer and context.
        """
        
        try:
            with open(TEST_QUESTIONS_PATH, 'r') as file:
                labeled_data = json.load(file)
            # return [labeled_data[0], labeled_data[1]]
            return labeled_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading test questions: {e}")
            return []
    
    @staticmethod
    def find_chunk_containing_context(retrieved_chunks, context):
        """
        Find the first chunk that contains the specified context.
        
        Args:
            retrieved_chunks: List of text chunks to search through
            context: The context text to look for
            
        Returns:
            tuple: (index, chunk) of the first matching chunk, or (None, None) if not found
        """
        # TODO: Implement a more efficient search algorithm if needed
        for i, chunk in enumerate(retrieved_chunks):
            if context in chunk:
                return i, chunk
        
        return -1, None
    
    