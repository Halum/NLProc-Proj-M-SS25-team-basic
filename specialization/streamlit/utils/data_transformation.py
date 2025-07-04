"""
Data transformation utilities for the RAG dashboard visualizations.

This module contains functions for preparing and transforming evaluation data
for visualization in the streamlit application. It consolidates data processing
operations from various parts of the application into a single module.
"""

import pandas as pd
import numpy as np
import logging
import statistics
from datetime import datetime
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add the project root to the path if needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import project configuration
from specialization.config.config import EVALUATION_INSIGHTS_PATH, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== General Utility Functions =====

def calculate_dynamic_chart_height(df: pd.DataFrame, query_col: str = 'Query', 
                                min_height: int = 600, 
                                base_multiplier: int = 40, 
                                text_factor: float = 0.8) -> int:
    """
    Calculate dynamic chart height based on number of entries and query text length.
    
    Args:
        df: DataFrame containing queries
        query_col: Name of the column containing query text
        min_height: Minimum height in pixels
        base_multiplier: Base pixels per row
        text_factor: Factor for additional height based on text length
    
    Returns:
        Calculated height in pixels
    """
    # Get number of entries
    num_entries = len(df)
    
    # Calculate average query text length if the query column exists
    if query_col in df.columns:
        text_lengths = [len(str(q)) for q in df[query_col]]
        if text_lengths:
            avg_length = statistics.mean(text_lengths)
            # Add height for longer than average queries
            text_adjustment = max(0, avg_length - 50) * text_factor
        else:
            text_adjustment = 0
    else:
        text_adjustment = 0
    
    # Calculate height with minimum value
    height = max(min_height, int(num_entries * (base_multiplier + text_adjustment)))
    
    return height

def extract_nested_score(record: Dict[str, Any], key_path: List[str], default: float = 0.0) -> float:
    """
    Safely extract a nested score from a record.
    
    Args:
        record: The record containing nested score data
        key_path: List of keys to traverse the nested structure
        default: Default value if the key doesn't exist
        
    Returns:
        The extracted score value or default value
    """
    value = record
    try:
        for key in key_path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

# ===== Single Insight Transformations =====

def prepare_correctness_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for answer correctness distribution visualization.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        DataFrame with counts of correct and incorrect answers
    """
    # Calculate correct vs incorrect counts
    correct_counts = insights_df['is_correct'].value_counts().reset_index()
    correct_counts.columns = ['Is Correct', 'Count']
    
    # Replace boolean values with readable labels
    correct_counts['Is Correct'] = correct_counts['Is Correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Ensure Count column is numeric
    correct_counts['Count'] = pd.to_numeric(correct_counts['Count'])
    
    return correct_counts

def prepare_similarity_distribution_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for similarity score distribution visualization.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        DataFrame with similarity scores and correctness information
    """
    # Process each record to get similarity data
    data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            
            # Get context analysis
            analysis = analyze_context_retrieval(record_dict)
            
            # Get similarity score
            avg_similarity = record_dict.get('avg_similarity_score')
            if avg_similarity is not None:
                data.append({
                    'Query': analysis['query'],
                    'Average Similarity': avg_similarity,
                    'Is Correct': 'Correct' if analysis['is_correct'] else 'Incorrect'
                })
    
    # Create dataframe from collected data
    plot_df = pd.DataFrame(data)
    
    # Make sure we have numeric values for similarity scores
    plot_df['Average Similarity'] = pd.to_numeric(plot_df['Average Similarity'], errors='coerce')
    
    # Drop any NaN values after conversion
    plot_df = plot_df.dropna(subset=['Average Similarity'])
    
    return plot_df

def prepare_bert_score_data(insights_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare BERT score data for visualization.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        Tuple of (DataFrame with BERT scores, boolean indicating if data is available)
    """
    has_bert_data = 'bert_score' in insights_df.columns and not insights_df['bert_score'].isna().all()
    
    if not has_bert_data:
        return pd.DataFrame(), False
    
    # Extract BERT scores
    bert_data = []
    for _, row in insights_df.iterrows():
        if isinstance(row['bert_score'], dict):
            bert_data.append({
                'Query': row['question'],
                'Precision': row['bert_score'].get('bert_precision', None),
                'Recall': row['bert_score'].get('bert_recall', None),
                'F1': row['bert_score'].get('bert_f1', None),
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect'
            })
    
    bert_df = pd.DataFrame(bert_data)
    
    # Ensure numeric data
    for col in ['Precision', 'Recall', 'F1']:
        bert_df[col] = pd.to_numeric(bert_df[col], errors='coerce')
    
    bert_df = bert_df.dropna(subset=['Precision', 'Recall', 'F1'])
    
    return bert_df, not bert_df.empty

def prepare_rouge_score_data(insights_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare ROUGE score data for visualization.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        Tuple of (DataFrame with ROUGE scores, boolean indicating if data is available)
    """
    has_rouge_data = 'rouge_score' in insights_df.columns and not insights_df['rouge_score'].isna().all()
    
    if not has_rouge_data:
        return pd.DataFrame(), False
    
    # Extract ROUGE scores
    rouge_data = []
    for _, row in insights_df.iterrows():
        if isinstance(row['rouge_score'], dict):
            rouge_data.append({
                'Query': row['question'],
                'ROUGE-1': row['rouge_score'].get('rouge1_fmeasure', None),
                'ROUGE-2': row['rouge_score'].get('rouge2_fmeasure', None),
                'ROUGE-L': row['rouge_score'].get('rougeL_fmeasure', None),
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect'
            })
    
    rouge_df = pd.DataFrame(rouge_data)
    
    # Ensure numeric data
    for col in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        rouge_df[col] = pd.to_numeric(rouge_df[col], errors='coerce')
    
    rouge_df = rouge_df.dropna(subset=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    
    return rouge_df, not rouge_df.empty

# Old calculate_context_retrieval_position function has been replaced by analyze_context_retrieval

def prepare_gold_context_presence_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for gold context presence visualization.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        DataFrame with gold context retrieval positions and correctness information
    """
    # Process each record to find context positions
    context_data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            # Convert Series to dict for processing
            record_dict = record.to_dict()
            analysis = analyze_context_retrieval(record_dict)
        elif isinstance(record, dict):
            analysis = analyze_context_retrieval(record)
        else:
            continue
        
        # Only include records where a position was actually found
        if analysis['position'] is not None:
            context_data.append({
                'Query': analysis['query'],
                'Position': analysis['position'],
                'Is Correct': 'Correct' if analysis['is_correct'] else 'Incorrect'
            })
    
    context_df = pd.DataFrame(context_data)
    return context_df

def calculate_overall_metrics(insights_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall performance metrics for the RAG system.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    total_queries = len(insights_df)
    correct_answers = insights_df['is_correct'].sum()
    correct_percent = (correct_answers / total_queries) * 100 if total_queries > 0 else 0
    
    metrics['total_queries'] = total_queries
    metrics['correct_answers'] = int(correct_answers)
    metrics['accuracy_percent'] = correct_percent
    
    # Average similarity score
    metrics['avg_similarity'] = insights_df['avg_similarity_score'].mean()
    
    # Calculate average BERT score if available
    if 'bert_score' in insights_df.columns and not insights_df['bert_score'].isna().all():
        avg_bert_f1 = np.mean([
            score.get('bert_f1', 0) 
            for score in insights_df['bert_score'] 
            if isinstance(score, dict)
        ])
        metrics['avg_bert_f1'] = avg_bert_f1
    
    # Calculate average ROUGE score if available
    if 'rouge_score' in insights_df.columns and not insights_df['rouge_score'].isna().all():
        avg_rouge_f1 = np.mean([
            score.get('rouge1_fmeasure', 0) 
            for score in insights_df['rouge_score'] 
            if isinstance(score, dict)
        ])
        metrics['avg_rouge_f1'] = avg_rouge_f1
    
    # Calculate context retrieval metrics
    context_positions = []
    total_analyzed = 0
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            analysis = analyze_context_retrieval(record_dict)
            total_analyzed += 1
            if analysis['position']:
                context_positions.append(analysis['position'])
    
    if context_positions and total_analyzed > 0:
        metrics['context_found_percent'] = len(context_positions) / total_analyzed * 100
        metrics['avg_context_position'] = np.mean(context_positions)
    else:
        metrics['context_found_percent'] = 0
        metrics['avg_context_position'] = None
    
    return metrics

# ===== Historical Data Transformations =====

def load_all_insight_files() -> List[Tuple[str, datetime, List[Dict[str, Any]]]]:
    """
    Load all available evaluation insight files for trend analysis.
    
    Returns:
        List of tuples with (file_path, timestamp, insights_data)
    """
    try:
        # Get the insights directory - first try the configured path
        insights_dir = Path(EVALUATION_INSIGHTS_PATH).parent
        
        # If that doesn't exist, try to find it relative to the current file
        if not insights_dir.exists():
            logging.warning(f"Configured insights directory does not exist: {insights_dir}")
            # Try to find the directory relative to the project root
            project_root = Path(__file__).parent.parent.parent.parent
            alternate_insights_dir = project_root / "specialization" / "data" / "insight"
            
            if alternate_insights_dir.exists():
                logging.info(f"Using alternate insights directory: {alternate_insights_dir}")
                insights_dir = alternate_insights_dir
            else:
                logging.error(f"Could not find insights directory at: {alternate_insights_dir}")
                return []
        
        # Find all JSON files in the directory
        json_files = list(insights_dir.glob("**/*.json"))
        
        # Filter for evaluation insight files
        insight_files = [f for f in json_files if "evaluation" in f.name.lower()]
        
        if not insight_files:
            logging.warning(f"No evaluation insight files found in {insights_dir}")
            return []
        
        # Sort files by modification time (newest first)
        insight_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Load data from each file
        result = []
        
        for file_path in insight_files:
            try:
                # Extract timestamp from filename or fallback to file modification time
                timestamp = extract_timestamp_from_filename(file_path.name)
                if not timestamp:
                    timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Load JSON data
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if the file contains insight data
                if isinstance(data, list) and len(data) > 0:
                    # Check if the first item looks like an insight record
                    if isinstance(data[0], dict) and 'question' in data[0]:
                        result.append((str(file_path), timestamp, data))
                    else:
                        # Skip files that don't contain valid insight records
                        pass
                else:
                    # Skip files that don't contain a list of insights
                    pass
            
            except Exception as e:
                logging.error(f"Error loading insight file {file_path}: {e}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error loading insight files: {e}")
        return []

def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from insight filename.
    Expected formats:
    - evaluation_insights_YYYYMMDD_HHMMSS.json
    - evaluation_insights_YYYYMMDD.json
    
    Args:
        filename: Name of the insight file
        
    Returns:
        Extracted timestamp or None if not found
    """
    try:
        # Try to match the format with time
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        
        # Try to match the format with date only
        match = re.search(r'(\d{8})', filename)
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, "%Y%m%d")
        
        return None
        
    except Exception:
        return None

def calculate_average_metrics(insight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate average metrics from a list of insight records.
    
    Args:
        insight_data: List of insight records
        
    Returns:
        Dictionary of average metrics
    """
    if not insight_data:
        return {}
    
    # Convert to DataFrame for easier analysis
    try:
        df = pd.DataFrame(insight_data)
        
        # Basic metrics
        total_samples = len(df)
        correct_count = df['is_correct'].sum() if 'is_correct' in df.columns else 0
        
        metrics = {
            'total_samples': total_samples,
            'correct_count': correct_count
        }
        
        if total_samples > 0:
            # Accuracy percentage
            metrics['accuracy_percent'] = (correct_count / total_samples) * 100
            
            # Average similarity score
            if 'avg_similarity_score' in df.columns:
                metrics['avg_similarity'] = df['avg_similarity_score'].mean()
            
            # BERT scores
            if 'bert_score' in df.columns:
                bert_precision = []
                bert_recall = []
                bert_f1 = []
                
                for _, row in df.iterrows():
                    if isinstance(row['bert_score'], dict):
                        if 'bert_precision' in row['bert_score']:
                            bert_precision.append(row['bert_score']['bert_precision'])
                        if 'bert_recall' in row['bert_score']:
                            bert_recall.append(row['bert_score']['bert_recall'])
                        if 'bert_f1' in row['bert_score']:
                            bert_f1.append(row['bert_score']['bert_f1'])
                
                if bert_precision:
                    metrics['avg_bert_precision'] = np.mean(bert_precision)
                if bert_recall:
                    metrics['avg_bert_recall'] = np.mean(bert_recall)
                if bert_f1:
                    metrics['avg_bert_f1'] = np.mean(bert_f1)
            
            # ROUGE scores
            if 'rouge_score' in df.columns:
                rouge1 = []
                rouge2 = []
                rougeL = []
                
                for _, row in df.iterrows():
                    if isinstance(row['rouge_score'], dict):
                        if 'rouge1_fmeasure' in row['rouge_score']:
                            rouge1.append(row['rouge_score']['rouge1_fmeasure'])
                        if 'rouge2_fmeasure' in row['rouge_score']:
                            rouge2.append(row['rouge_score']['rouge2_fmeasure'])
                        if 'rougeL_fmeasure' in row['rouge_score']:
                            rougeL.append(row['rouge_score']['rougeL_fmeasure'])
                
                if rouge1:
                    metrics['avg_rouge1_f1'] = np.mean(rouge1)
                if rouge2:
                    metrics['avg_rouge2_f1'] = np.mean(rouge2)
                if rougeL:
                    metrics['avg_rougeL_f1'] = np.mean(rougeL)
            
            # Context distance (retrieval position)
            context_positions = []
            total_analyzed = 0
            for _, row in df.iterrows():
                analysis = analyze_context_retrieval(row.to_dict())
                total_analyzed += 1
                if analysis['position']:
                    context_positions.append(analysis['position'])
            
            if context_positions and total_analyzed > 0:
                metrics['avg_context_distance'] = np.mean(context_positions)
                metrics['context_found_percent'] = len(context_positions) / total_analyzed * 100
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating average metrics: {e}")
        return {}

def prepare_historical_metrics() -> pd.DataFrame:
    """
    Process all insight files and extract average metrics for trending.
    
    Returns:
        DataFrame with historical metrics
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
            except Exception:
                # If conversion fails, we'll leave it out
                pass
            
            historical_data.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    except Exception as e:
        logging.error(f"Error preparing historical metrics: {e}")
        return pd.DataFrame()

def extract_score_distributions(historical_data: pd.DataFrame, score_type: str) -> Tuple[List[List[float]], List[str]]:
    """
    Extract score distributions for box plots from historical data.
    
    Args:
        historical_data: DataFrame containing historical metrics with raw insight data
        score_type: Type of score to extract ('bert', 'rouge', 'similarity')
    
    Returns:
        Tuple of (list of score lists, iteration labels)
    """
    # Create iteration labels with sample info
    iteration_labels = []
    
    # Create labels for iterations
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
    
    # Check if we have raw insight data
    has_insight_data = all('insight_data' in row and isinstance(row['insight_data'], pd.DataFrame) 
                         for _, row in historical_data.iterrows())
    
    if not has_insight_data:
        return [], iteration_labels
    
    # Prepare data based on score type
    if score_type == 'bert':
        precision_data = []
        recall_data = []
        f1_data = []
        
        for _, row in historical_data.iterrows():
            insightdf = row['insight_data']
            
            # Extract BERT scores from each insight
            iter_precision = []
            iter_recall = []
            iter_f1 = []
            
            for _, insight in insightdf.iterrows():
                if 'bert_score' in insight and isinstance(insight['bert_score'], dict):
                    if 'bert_precision' in insight['bert_score']:
                        iter_precision.append(insight['bert_score']['bert_precision'])
                    if 'bert_recall' in insight['bert_score']:
                        iter_recall.append(insight['bert_score']['bert_recall'])
                    if 'bert_f1' in insight['bert_score']:
                        iter_f1.append(insight['bert_score']['bert_f1'])
            
            precision_data.append(iter_precision)
            recall_data.append(iter_recall)
            f1_data.append(iter_f1)
        
        return [precision_data, recall_data, f1_data], iteration_labels
    
    elif score_type == 'rouge':
        rouge1_data = []
        rouge2_data = []
        rougeL_data = []
        
        for _, row in historical_data.iterrows():
            insightdf = row['insight_data']
            
            # Extract ROUGE scores from each insight
            iter_rouge1 = []
            iter_rouge2 = []
            iter_rougeL = []
            
            for _, insight in insightdf.iterrows():
                if 'rouge_score' in insight and isinstance(insight['rouge_score'], dict):
                    if 'rouge1_fmeasure' in insight['rouge_score']:
                        iter_rouge1.append(insight['rouge_score']['rouge1_fmeasure'])
                    if 'rouge2_fmeasure' in insight['rouge_score']:
                        iter_rouge2.append(insight['rouge_score']['rouge2_fmeasure'])
                    if 'rougeL_fmeasure' in insight['rouge_score']:
                        iter_rougeL.append(insight['rouge_score']['rougeL_fmeasure'])
            
            rouge1_data.append(iter_rouge1)
            rouge2_data.append(iter_rouge2)
            rougeL_data.append(iter_rougeL)
        
        return [rouge1_data, rouge2_data, rougeL_data], iteration_labels
    
    elif score_type == 'similarity':
        similarity_data = []
        
        for _, row in historical_data.iterrows():
            insightdf = row['insight_data']
            
            # Extract similarity scores from each insight
            iter_similarity = []
            
            for _, insight in insightdf.iterrows():
                if 'avg_similarity_score' in insight:
                    iter_similarity.append(insight['avg_similarity_score'])
            
            similarity_data.append(iter_similarity)
        
        return [similarity_data], iteration_labels
    
    # The customer did not request context position distribution charts, this is a simplified version
    # that returns minimal data needed for historical context metrics without the additional distributions
    elif score_type == 'context':
        position_data = []
        
        for _, row in historical_data.iterrows():
            insightdf = row['insight_data']
            
            # Extract context positions from each insight
            iter_positions = []
            
            for _, insight in insightdf.iterrows():
                # Use our single source of truth function
                analysis = analyze_context_retrieval(insight.to_dict())
                
                # Track position if context was found
                if analysis['position'] is not None:
                    iter_positions.append(analysis['position'])
            
            position_data.append(iter_positions)
        
        return [position_data], iteration_labels
    
    else:
        return [], iteration_labels

def analyze_context_retrieval(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze context retrieval for a single record, calculating position and correctness.
    This function serves as a single source of truth for context analysis.
    
    Args:
        record: A single evaluation record
        
    Returns:
        Dictionary with analysis results containing:
        - position: Position of gold context (1-indexed) or None if not found
        - is_correct: Whether the answer was correct
        - query: The original query
    """
    result = {
        'position': None,
        'is_correct': record.get('is_correct', False),
        'query': record.get('question', 'Unknown query')
    }
    
    # Skip if either is missing
    if not ('gold_context' in record and 'context' in record):
        return result
        
    gold_context = record['gold_context']
    retrieved_contexts = record['context']
    
    # Skip if either is missing or contexts is not a list
    if not (gold_context and retrieved_contexts and isinstance(retrieved_contexts, list)):
        return result
    
    # Extract content from gold context based on its structure
    gold_content = None
    
    # Handle different formats of gold_context
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
    if not gold_content:
        return result
    
    # Normalize gold content to improve matching
    gold_content_norm = gold_content.strip().lower()
    if len(gold_content_norm) < 10:  # Skip very short contexts as they may cause false positives
        return result
        
    for i, ctx in enumerate(retrieved_contexts[:5]):  # Check only top 5 contexts
        if isinstance(ctx, dict) and 'content' in ctx:
            ctx_content = ctx['content'].strip().lower()
            # Check for substantial overlap, not just exact substring match
            if gold_content_norm in ctx_content or ctx_content in gold_content_norm:
                result['position'] = i + 1  # Convert to 1-indexed position
                break
        elif isinstance(ctx, str):
            ctx_content = str(ctx).strip().lower()
            if gold_content_norm in ctx_content or ctx_content in gold_content_norm:
                result['position'] = i + 1  # Convert to 1-indexed position
                break
    
    return result
