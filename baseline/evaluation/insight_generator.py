import pandas as pd

from config.config import INSIGHT_FOLDER_PATH
from postprocessor.document_writer import DocumentWriter

class InsightGenerator:
    def __init__(self):
        """
        Initialize the InsightGenerator class.
        """
        self.insight_df = pd.DataFrame(columns=["chunk_strategy", "number_of_chunks", "retrieved_chunk_rank", "correct_answer"])
        
    def update_insight(self, chunk_strategy, number_of_chunks, retrieved_chunk_rank, correct_answer, similarity_scores):
        """
        Update the insight DataFrame with the results of the query.
        Args:
            chunk_strategy (str): The chunking strategy used.
            number_of_chunks (int): The number of chunks generated.
            retrieved_chunk_rank (int): The rank of the retrieved chunk.
            correct_answer (bool): Whether the answer was correct or not.
            similarity_scores (list): A list of similarity scores for the chunks.
        Returns:
            pd.DataFrame: The updated DataFrame.
        """
        # Update the DataFrame with the results using pd.concat instead of append
        new_row = pd.DataFrame({
            "chunk_strategy": [chunk_strategy],
            "number_of_chunks": [number_of_chunks],
            "retrieved_chunk_rank": [retrieved_chunk_rank],
            "correct_answer": [correct_answer],
            "similarity_scores": [similarity_scores]
        })
        self.insight_df = pd.concat([self.insight_df, new_row], ignore_index=True)
        
    def save_insight(self, file_name):
        """
        Save the insight DataFrame to a CSV file.
        Args:
            file_name (str): The name to save the CSV file.
        """
        DocumentWriter.df_to_csv(self.insight_df, INSIGHT_FOLDER_PATH, file_name)
        
    @staticmethod
    def human_feedback(expected_answer, generated_answer):
        """
        Function to simulate human feedback on the generated answer.
        
        Args:
            expected_answer (str): The expected answer.
            generated_answer (str): The generated answer.
            
        Returns:
            bool: True if the generated answer is correct, False otherwise.
        """
        print("Is the answer correct? (y/n)")
        feedback = input().strip().lower()
        
        return feedback == 'y'