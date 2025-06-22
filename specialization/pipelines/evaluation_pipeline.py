#!/usr/bin/env python3
"""
Evaluation Pipeline for Specialization Track

This pipeline runs evaluation on a set of gold standard inputs by:
1. Loading existing embeddings from ChromaDB
2. Reading gold standard input data from a JSON file
3. Parsing each query to extract metadata filters using QueryParser
4. Running each query through the RAG system with the parsed query and filters
5. Comparing generated answers to gold standard answers
6. Generating insights and metrics for performance evaluation
7. Saving results for later analysis

Features:
- Batch processing of evaluation data
- Reuse of UserQueryPipeline components and QueryParser for consistent behavior
- Intelligent query parsing and metadata filter extraction
- Integration with specialized insight generation
- Performance metrics collection and analysis
- Automated evaluation of answer correctness
- Detailed logging of parsed queries, filters, and similarity scores
"""

from math import log
import os
import sys
import json
import logging
from typing import List, Dict, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from specialization.pipelines.user_query import UserQueryPipeline
from specialization.evaluation import InsightGenerator
from specialization.evaluation import MetricsGenerator
from specialization.generator.query_parser import QueryParser
from specialization.config.config import (
    GOLD_INPUT_PATH, 
    EVALUATION_INSIGHTS_PATH,
    LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set log level based on configuration
logger.setLevel(getattr(logging, LOG_LEVEL))


class EvaluationPipeline:
    """
    Pipeline for evaluating RAG system against gold standard data.
    """
    
    def __init__(self, 
                gold_data_path: str,
                insight_path: str,
                chunk_size: int = None,
                use_existing_db: bool = True):
        """
        Initialize the evaluation pipeline.
        
        Args:
            gold_data_path (str): Path to gold standard data
            insights_path (str): Path to save insights
            chunk_size (int): Chunk size for retrieval
            use_existing_db (bool): Whether to use existing DB
        """
        self.gold_data_path = gold_data_path
        self.insights_path = insight_path
        
        # Initialize user query pipeline for RAG functionality
        self.query_pipeline = UserQueryPipeline(
            chunk_size=chunk_size, 
            use_existing_db=use_existing_db
        )
        
        # Initialize query parser for consistent query parsing
        self.query_parser = QueryParser()
        
        # Initialize insights generator
        self.insight_generator = InsightGenerator(insight_path=insight_path)
        
        logger.info("Evaluation Pipeline initialized successfully")
    
    def load_gold_data(self) -> List[Dict]:
        """
        Load gold standard data from JSON file.
        
        Returns:
            List[Dict]: Gold standard data
        """
        try:
            with open(self.gold_data_path, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)
            
            logger.info(f"Loaded {len(gold_data)} gold standard queries from {self.gold_data_path}")
            return gold_data
        except Exception as e:
            logger.error(f"Error loading gold data: {e}")
            raise
    
    def evaluate_single_query(self, gold_item: Dict) -> Dict:
        """
        Evaluate a single gold standard query.
        
        Args:
            gold_item (Dict): Gold standard item with question and answer
            
        Returns:
            Dict: Evaluation results
        """
        question = gold_item['question']
        gold_answer = gold_item['answer']
        
        logger.info(f"Evaluating question: {question}")
        
        # Parse query to extract filters and clean question - same as user_query.py
        parsed_query, parsed_filters = self.query_parser.parse_with_chroma_filters(question)
        
        logger.info(f"Parsed query: {parsed_query}")
        logger.info(f"Parsed filters: {parsed_filters}")
        
        # Run query through pipeline with parsed query and filters
        result = self.query_pipeline.run_single_query(
            query=parsed_query,
            filter_dict=parsed_filters,
            show_context=False
        )
        
        generated_answer = result['answer']
        context = result['context']
        
        # Determine if answer is correct through automatic evaluation
        # Simple substring match - could be enhanced with NLP techniques
        is_correct = gold_answer.lower() in generated_answer.lower()
        
        # Extract the average similarity score from the context if available
        avg_similarity_score = None
        if context and len(context) > 0 and all('score' in doc for doc in context):
            avg_similarity_score = sum(doc['score'] for doc in context) / len(context)
        # Create insight data dictionary
        insight_data = {
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "context": context,
            "is_correct": is_correct,
            "avg_similarity_score": avg_similarity_score,
            "metadata_filters": parsed_filters,
            "parsed_query": parsed_query,
            "bert_score": MetricsGenerator.calculate_bert_score(gold_answer, generated_answer),
            "rouge_score": MetricsGenerator.calculate_rouge_score(gold_answer, generated_answer)
        }
        # Add to insights
        self.insight_generator.update_insight(**insight_data)
        
        return {
            'question': question,
            'gold_answer': gold_answer,
            'generated_answer': generated_answer,
            'is_correct': is_correct
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation on all gold standard queries.
        
        Returns:
            Dict: Evaluation metrics and results
        """
        gold_data = self.load_gold_data()
        results = []
        
        for i, gold_item in enumerate(gold_data):
            logger.info(f"Processing query {i+1}/{len(gold_data)}")
            result = self.evaluate_single_query(gold_item)
            results.append(result)
        
        # Use InsightGenerator's calculate_metrics method instead of manual calculation
        metrics = self.insight_generator.calculate_accuracy_metrics()
        
        # Add the detailed results to the metrics
        metrics['results'] = results
        
        # Final save of insights after all evaluations are complete
        self.insight_generator.save_insights()
        
        logger.info(f"Evaluation complete with accuracy: {metrics['accuracy']:.2f}")
        
        return metrics


def main():
    """Main function to run the evaluation pipeline."""
    try:
        # Initialize and run evaluation pipeline
        pipeline = EvaluationPipeline(
            gold_data_path=GOLD_INPUT_PATH,
            insight_path=EVALUATION_INSIGHTS_PATH,
            use_existing_db=True
        )
        
        metrics = pipeline.run_evaluation()
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total queries: {metrics['total_queries']}")
        print(f"Correct answers: {metrics['correct_answers']}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print("-"*80)
        print("Query parsing was used for all evaluation queries")
        print(f"Evaluation insights saved to: {EVALUATION_INSIGHTS_PATH}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    main()
