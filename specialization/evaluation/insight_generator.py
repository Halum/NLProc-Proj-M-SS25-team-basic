"""
Module for generating insights from evaluation results.
This file contains the InsightGenerator class that provides:
1. Metrics collection and aggregation for system performance analysis
2. Data persistence for evaluation results and performance measurements
3. Structured logging of query, context, and answer performance

Based on the baseline insight generator but specialized for movie data retrieval.
"""

import pandas as pd
import os
import logging
from datetime import datetime

# Import the document writer from baseline to maintain consistency
from baseline.postprocessor.document_writer import DocumentWriter
from specialization.config.config import APPEND_INSIGHTS, LOG_LEVEL

# Setup logging
logger = logging.getLogger(__name__)

# Set log level based on configuration
logger.setLevel(getattr(logging, LOG_LEVEL))

class InsightGenerator:
    """
    Class for generating and saving insights from evaluation results.
    Specialized for movie RAG system evaluation.
    """
    
    def __init__(self, insight_path, flush_threshold=5):
        """
        Initialize the InsightGenerator class.
        
        Args:
            insight_path (str): Path to save insight files
            flush_threshold (int): Number of insights before auto-saving
        """
        self.create_new_insights = not APPEND_INSIGHTS
        self.flush_threshold = flush_threshold
        self.insight_df = pd.DataFrame(columns=[
            "question", 
            "parsed_question",
            "gold_answer",
            "generated_answer", 
            "is_correct",
            "metadata_filters", 
            "avg_similarity_score", 
            "bert_score",
            "rouge_score",
            "timestamp",
            "gold_context",
            "context", 
        ])
        # this copy will be used to generate metrics in calculate_metrics
        self.insight_df_copy = self.insight_df.copy()
        
        self.insight_dir, self.insight_path = self.generate_timestamped_filename(insight_path)
        
        logger.info(f"InsightGenerator initialized with path: {self.insight_path}")
        
    def generate_timestamped_filename(self, insight_path):
        # Extract directory and base filename from the path
        insight_dir = os.path.dirname(insight_path)
        insight_filename = os.path.basename(insight_path)
        
        # Create a timestamped filename by adding timestamp before extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = os.path.splitext(insight_filename)
        timestamped_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
        
        # Full path to the timestamped file
        timestamped_insight_path = os.path.join(insight_dir, timestamped_filename)

        return insight_dir, timestamped_insight_path

    def update_insight(self, question, gold_answer, generated_answer, context,
                      is_correct, avg_similarity_score=None, metadata_filters=None,
                      parsed_question=None, bert_score=None, rouge_score=None,
                      gold_context=None):
        """
        Update the insight DataFrame with the results of the query.
        
        Args:
            question (str): The query question
            gold_answer (str): Expected gold standard answer
            generated_answer (str): Answer generated by the model
            context (list): Retrieved chunks/context used for generation
            is_correct (bool): Whether the answer matches gold standard
            avg_similarity_score (float, optional): Average similarity score of retrieved documents
            metadata_filters (dict, optional): Any filters applied during retrieval
            parsed_question (str, optional): The query after parsing and cleaning
            bert_score (dict, optional): Dictionary with bert precision, recall and f1 scores
            rouge_score (dict, optional): Dictionary with rouge precision, recall and f1 scores
            gold_context (list, optional): The gold standard context for the question

        Returns:
            pd.DataFrame: The updated DataFrame
        """            
        # Prepare basic row data
        row_data = {
            "question": [question],
            "parsed_question": [parsed_question],
            "gold_answer": [gold_answer],
            "generated_answer": [generated_answer],
            "is_correct": [is_correct],
            "metadata_filters": [metadata_filters],
            "avg_similarity_score": [avg_similarity_score],
            "bert_score": [bert_score],
            "rouge_score": [rouge_score],
            "timestamp": pd.Timestamp.now(),
            "context": [context],
            "gold_context": [gold_context],
        }
        
        # Update the DataFrame with the results
        new_row = pd.DataFrame(row_data)
        
        self.insight_df = pd.concat([self.insight_df, new_row], ignore_index=True)
        self.insight_df_copy = pd.concat([self.insight_df_copy, new_row], ignore_index=True)
        
        if len(self.insight_df) >= self.flush_threshold:
            logger.info("Flushing insights to disk...")
            self.save_insights()
            
        return self.insight_df

    def save_insights(self):
        """
        Save the insight DataFrame using the DocumentWriter.
        Creates necessary directories if they don't exist.
        Adds timestamp to the filename for historical tracking.
        """
        if len(self.insight_df) == 0:
            logger.info("No insights to save")
            return
            
        try:            
            # Create directory if it doesn't exist
            os.makedirs(self.insight_dir, exist_ok=True)
            
            # Get just the filename from the full path
            filename = os.path.basename(self.insight_path)
            
            # Use DocumentWriter to save insights with just the filename
            DocumentWriter.df_to_json(self.insight_df, self.insight_dir, filename, append=not self.create_new_insights )
            self.create_new_insights = False
            
            # Clear the DataFrame after saving
            self.insight_df = self.insight_df.iloc[0:0]

            logger.info(f"Insights saved to {self.insight_path}")

        except Exception as e:
            logger.error(f"Error saving insights: {e}")
            raise
            
    def calculate_accuracy_metrics(self):
        """
        Calculate evaluation metrics from collected insights.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if len(self.insight_df_copy) == 0:
            return {"error": "No insights available for metric calculation"}
            
        metrics = {
            "total_queries": len(self.insight_df_copy),
            "correct_answers": sum(self.insight_df_copy["is_correct"]),
            "accuracy": sum(self.insight_df_copy["is_correct"]) / len(self.insight_df_copy),
        }
        
        return metrics
