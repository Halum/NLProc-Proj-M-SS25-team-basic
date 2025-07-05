"""
Historical metrics and trend analysis for RAG evaluation.

This module contains functions for loading and analyzing historical evaluation data
to track system performance over time.
"""
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging

# Import project configuration
from specialization.config.config import EVALUATION_INSIGHTS_PATH, LOG_LEVEL
from .context_metrics import analyze_context_retrieval

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from insight filename.
    Expected formats:
    - evaluation_insights_YYYYMMDD_HHMMSS.json
    - evaluation_insights_YYYYMMDD.json
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

def load_all_insight_files() -> List[Tuple[str, datetime, List[Dict[str, Any]]]]:
    """
    Load all available evaluation insight files for trend analysis.
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
                    result.append((str(file_path), timestamp, data))
                else:
                    logging.warning(f"Skipping file with invalid format: {file_path}")
            
            except Exception as e:
                logging.error(f"Error loading insight file {file_path}: {e}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error loading insight files: {e}")
        return []

def calculate_average_metrics(insight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate average metrics from a list of insight records.
    """
    if not insight_data:
        return {}
    
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
            
            # Context metrics
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
            
            # Store the original insight data as a DataFrame for potential recalculation
            try:
                metrics['insight_data'] = pd.DataFrame(insight_data)
            except Exception:
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
    """
    # Create iteration labels with sample info
    iteration_labels = []
    
    # Create labels for iterations
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
    
    # Check if we have raw insight data
    has_insight_data = all('insight_data' in row and isinstance(row['insight_data'], pd.DataFrame) 
                         for _, row in historical_data.iterrows())
    
    if not has_insight_data:
        return [], iteration_labels
    
    distributions = []
    
    for _, row in historical_data.iterrows():
        if score_type == 'bert':
            scores = [rec['bert_score'].get('bert_f1', 0) for _, rec in row['insight_data'].iterrows()
                     if isinstance(rec.get('bert_score', None), dict)]
        elif score_type == 'rouge':
            scores = [rec['rouge_score'].get('rouge1_fmeasure', 0) for _, rec in row['insight_data'].iterrows()
                     if isinstance(rec.get('rouge_score', None), dict)]
        elif score_type == 'similarity':
            scores = row['insight_data']['avg_similarity_score'].dropna().tolist()
        elif score_type == 'context':
            scores = [analyze_context_retrieval(rec.to_dict())['position'] 
                     for _, rec in row['insight_data'].iterrows()
                     if analyze_context_retrieval(rec.to_dict())['position'] is not None]
        else:
            scores = []
            
        distributions.append(scores)
    
    return distributions, iteration_labels
