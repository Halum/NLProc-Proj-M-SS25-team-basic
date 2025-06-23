"""
Utility functions for loading and processing historical data from insight files.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os
import re

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import project configuration
from specialization.config.config import EVALUATION_INSIGHTS_PATH, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_all_insight_files():
    """
    Loads all timestamped insight files in the insight directory.
    
    Returns:
        list: List of tuples containing (file_path, timestamp, insight_data)
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Get the insight directory
        insight_dir = os.path.dirname(project_root / EVALUATION_INSIGHTS_PATH)
        insight_base_name = os.path.splitext(os.path.basename(EVALUATION_INSIGHTS_PATH))[0]
        
        # Get all JSON files in the directory that match the pattern - only with timestamps
        pattern = re.compile(f"{insight_base_name}_(\\d{{8}}_\\d{{6}})\\.json$")
        insight_files = []
        
        if os.path.exists(insight_dir):
            for filename in os.listdir(insight_dir):
                if filename.endswith('.json'):
                    match = pattern.match(filename)
                    if match:
                        # Parse the timestamp from filename
                        timestamp_str = match.group(1)
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        file_path = os.path.join(insight_dir, filename)
                        
                        # Load the data
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                insight_data = json.load(f)
                                insight_files.append((file_path, timestamp, insight_data))
                        except Exception as e:
                            logging.error(f"Error loading file {file_path}: {e}")
        
        # Sort files by timestamp (oldest first)
        insight_files.sort(key=lambda x: x[1])
        
        logging.info(f"Loaded {len(insight_files)} insight files")
        return insight_files
    
    except Exception as e:
        logging.error(f"Error loading insight files: {e}")
        return []

def calculate_average_metrics(insight_data):
    """
    Calculate average metrics from a single insight file.
    
    Args:
        insight_data (list): List of insight records
        
    Returns:
        dict: Dictionary containing average metrics
    """
    try:
        # Initialize counters
        metrics = {
            'avg_similarity': 0.0,
            'avg_bert_precision': 0.0,
            'avg_bert_recall': 0.0,
            'avg_bert_f1': 0.0,
            'avg_rouge1_f1': 0.0,
            'avg_rouge2_f1': 0.0,
            'avg_rougeL_f1': 0.0,
            'count': 0,
            'correct_count': 0,
            'incorrect_count': 0,
            'total_samples': 0
        }
        
        # Initialize counters for each metric category
        bert_count = 0
        rouge_count = 0
        similarity_count = 0
        
        # Sum up metrics
        for record in insight_data:
            # Similarity score
            if 'avg_similarity_score' in record:
                metrics['avg_similarity'] += record['avg_similarity_score']
                similarity_count += 1
            
            # BERT scores
            if 'bert_score' in record and isinstance(record['bert_score'], dict):
                if 'bert_precision' in record['bert_score']:
                    metrics['avg_bert_precision'] += record['bert_score']['bert_precision']
                if 'bert_recall' in record['bert_score']:
                    metrics['avg_bert_recall'] += record['bert_score']['bert_recall']
                if 'bert_f1' in record['bert_score']:
                    metrics['avg_bert_f1'] += record['bert_score']['bert_f1']
                bert_count += 1
            
            # ROUGE scores
            if 'rouge_score' in record and isinstance(record['rouge_score'], dict):
                if 'rouge1_fmeasure' in record['rouge_score']:
                    metrics['avg_rouge1_f1'] += record['rouge_score']['rouge1_fmeasure']
                if 'rouge2_fmeasure' in record['rouge_score']:
                    metrics['avg_rouge2_f1'] += record['rouge_score']['rouge2_fmeasure']
                if 'rougeL_fmeasure' in record['rouge_score']:
                    metrics['avg_rougeL_f1'] += record['rouge_score']['rougeL_fmeasure']
                rouge_count += 1
            
            # Count correct/incorrect answers
            metrics['total_samples'] += 1
            if 'is_correct' in record:
                if record['is_correct']:
                    metrics['correct_count'] += 1
                else:
                    metrics['incorrect_count'] += 1
                
            metrics['count'] += 1
        
        # Calculate averages
        if similarity_count > 0:
            metrics['avg_similarity'] /= similarity_count
        
        if bert_count > 0:
            metrics['avg_bert_precision'] /= bert_count
            metrics['avg_bert_recall'] /= bert_count 
            metrics['avg_bert_f1'] /= bert_count
        
        if rouge_count > 0:
            metrics['avg_rouge1_f1'] /= rouge_count
            metrics['avg_rouge2_f1'] /= rouge_count
            metrics['avg_rougeL_f1'] /= rouge_count
            
        return metrics
    
    except Exception as e:
        logging.error(f"Error calculating average metrics: {e}")
        return {
            'avg_similarity': 0.0,
            'avg_bert_precision': 0.0,
            'avg_bert_recall': 0.0,
            'avg_bert_f1': 0.0,
            'avg_rouge1_f1': 0.0,
            'avg_rouge2_f1': 0.0,
            'avg_rougeL_f1': 0.0,
            'count': 0
        }

def get_historical_metrics():
    """
    Process all insight files and extract average metrics for trending.
    
    Returns:
        pd.DataFrame: DataFrame with historical metrics
    """
    try:
        # Get all insight files
        insight_files = load_all_insight_files()
        
        if not insight_files:
            logging.warning("No insight files found")
            return pd.DataFrame()
        
        # Process each file
        historical_data = []
        
        for file_path, timestamp, insight_data in insight_files:
            # Calculate average metrics for this file
            metrics = calculate_average_metrics(insight_data)
            
            # Add timestamp and record count
            metrics['timestamp'] = timestamp
            metrics['file_path'] = file_path
            metrics['record_count'] = len(insight_data)
            
            historical_data.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    except Exception as e:
        logging.error(f"Error getting historical metrics: {e}")
        return pd.DataFrame()
