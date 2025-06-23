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
            'avg_context_distance': 0.0,  # Added this for retrieval accuracy
            'correct_count': 0,
            'incorrect_count': 0,
            'total_samples': 0
        }
        
        # Initialize counters for each metric category
        bert_count = 0
        rouge_count = 0
        similarity_count = 0
        context_distance_count = 0  # Added for retrieval accuracy
        
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
            
            # Calculate context distance (retrieval position)
            # Find the position of gold_context in the context list
            if 'gold_context' in record and 'context' in record:
                gold_context = record['gold_context']
                retrieved_contexts = record['context']
                
                # Skip if either is missing
                if gold_context and retrieved_contexts and isinstance(retrieved_contexts, list):
                    # Initialize as not found
                    found = False
                    position = -1
                    
                    # We need to handle different formats of gold_context
                    gold_content = None
                    
                    # Extract content from gold context based on its structure
                    if isinstance(gold_context, list):
                        # If gold_context is a list, join all contents
                        gold_content = " ".join([
                            ctx.get('content', '') if isinstance(ctx, dict) else str(ctx)
                            for ctx in gold_context
                        ])
                    elif isinstance(gold_context, dict):
                        # If gold_context is a dict, get the content field
                        gold_content = gold_context.get('content', '')
                    else:
                        # If gold_context is a string or other type
                        gold_content = str(gold_context)
                    
                    # Check if gold context is present in any of the retrieved contexts
                    if isinstance(retrieved_contexts, list):
                        for i, ctx in enumerate(retrieved_contexts[:5]):  # Check only top 5 contexts
                            if isinstance(ctx, dict) and 'content' in ctx:
                                # Compare content
                                if gold_content and gold_content in ctx['content']:
                                    found = True
                                    position = i
                                    break
                            elif isinstance(ctx, str):
                                # Direct string comparison
                                if gold_content and gold_content in ctx:
                                    found = True
                                    position = i
                                    break
                    
                    # If we found the gold context, add its position to the metrics
                    if found:
                        # Add 1 to convert to 1-indexed position (positions start at 1, not 0)
                        metrics['avg_context_distance'] += (position + 1)
                        context_distance_count += 1
                        logging.debug(f"Found gold context at position {position+1}")
                    else:
                        logging.debug("Gold context not found in retrieved contexts")
            
            # Count correct/incorrect answers
            metrics['total_samples'] += 1
            if 'is_correct' in record:
                if record['is_correct']:
                    metrics['correct_count'] += 1
                else:
                    metrics['incorrect_count'] += 1
        
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
        
        # Add average context distance if we have data
        if context_distance_count > 0:
            metrics['avg_context_distance'] /= context_distance_count
        else:
            # If no context distance found, leave it as None to indicate missing data
            metrics['avg_context_distance'] = None
            
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
            'avg_context_distance': None  # Set to None to indicate missing data
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
            
            # Store the original insight data as a DataFrame for potential recalculation if needed
            try:
                metrics['insight_data'] = pd.DataFrame(insight_data)
            except:
                # If conversion fails, we'll leave it out
                pass
            
            historical_data.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    except Exception as e:
        logging.error(f"Error getting historical metrics: {e}")
        return pd.DataFrame()
