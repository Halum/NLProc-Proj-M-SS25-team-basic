"""
Basic metrics and data transformations for single insight analysis.

This module contains functions for preparing and transforming evaluation data
for basic visualizations in the streamlit application.
"""
from datetime import datetime
from pathlib import Path
import statistics
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging

# Import project configuration
from specialization.config.config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def calculate_dynamic_chart_height(df: pd.DataFrame, query_col: str = 'Query', 
                                min_height: int = 600, 
                                base_multiplier: int = 40, 
                                text_factor: float = 0.8) -> int:
    """
    Calculate dynamic chart height based on number of entries and query text length.
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

def prepare_correctness_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for answer correctness distribution visualization.
    """
    # Calculate correct vs incorrect counts
    correct_counts = insights_df['is_correct'].value_counts().reset_index()
    correct_counts.columns = ['Is Correct', 'Count']
    
    # Replace boolean values with readable labels
    correct_counts['Is Correct'] = correct_counts['Is Correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Ensure Count column is numeric
    correct_counts['Count'] = pd.to_numeric(correct_counts['Count'])
    
    return correct_counts

def prepare_correctness_by_groups_data(insights_df: pd.DataFrame, group_by: str = 'difficulty') -> pd.DataFrame:
    """
    Prepare data for answer correctness by groups visualization.
    """
    if group_by not in ['difficulty', 'tags']:
        raise ValueError("group_by must be either 'difficulty' or 'tags'")
        
    # Prepare data for grouping
    data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            
            is_correct = record_dict.get('is_correct', False)
            correctness_label = 'Correct' if is_correct else 'Incorrect'
            
            # Handle the grouping based on the specified field
            if group_by == 'difficulty':
                group = record_dict.get('difficulty', 'Unknown')
            else:  # tags
                tags = record_dict.get('tags', [])
                if isinstance(tags, list):
                    for tag in tags:
                        data.append({
                            'Group': tag,
                            'Correctness': correctness_label,
                            'Count': 1
                        })
                    continue
                else:
                    group = 'Untagged'
            
            data.append({
                'Group': group,
                'Correctness': correctness_label,
                'Count': 1
            })

    return prepare_group_data(data)

def prepare_group_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper function to process grouped data."""
    df = pd.DataFrame(data)
    
    if df.empty:
        return pd.DataFrame(columns=['Group', 'Correctness', 'Count'])
    
    # Group by Group and Correctness, sum the counts
    grouped_df = df.groupby(['Group', 'Correctness'])['Count'].sum().reset_index()
    
    # Pivot and process data
    pivot_df = grouped_df.pivot(index='Group', columns='Correctness', values='Count').fillna(0)
    
    # Ensure both columns exist
    if 'Correct' not in pivot_df.columns:
        pivot_df['Correct'] = 0
    if 'Incorrect' not in pivot_df.columns:
        pivot_df['Incorrect'] = 0
    
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.sort_values('Correct', ascending=False)
    
    return pivot_df

def prepare_similarity_distribution_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for similarity score distribution visualization.
    """
    data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            similarity_score = record_dict.get('avg_similarity_score', None)
            is_correct = record_dict.get('is_correct', False)
            query = record_dict.get('question', '') or record_dict.get('query', 'Unknown Query')
            
            if similarity_score is not None:
                data.append({
                    'Query': query,
                    'Average Similarity': similarity_score,
                    'Is Correct': 'Correct' if is_correct else 'Incorrect'
                })
    
    plot_df = pd.DataFrame(data)
    plot_df['Average Similarity'] = pd.to_numeric(plot_df['Average Similarity'], errors='coerce')
    plot_df = plot_df.dropna(subset=['Average Similarity'])
    
    return plot_df

def prepare_bert_score_data(insights_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare BERT score data for visualization.
    """
    has_bert_data = 'bert_score' in insights_df.columns and not insights_df['bert_score'].isna().all()
    
    if not has_bert_data:
        return pd.DataFrame(), False
    
    bert_data = []
    for _, row in insights_df.iterrows():
        if isinstance(row['bert_score'], dict):
            query = row.get('question', '') or row.get('query', 'Unknown Query')
            bert_data.append({
                'Query': query,
                'Precision': row['bert_score'].get('bert_precision', 0),
                'Recall': row['bert_score'].get('bert_recall', 0),
                'F1': row['bert_score'].get('bert_f1', 0),
                'Is Correct': 'Correct' if row.get('is_correct', False) else 'Incorrect'
            })
    
    bert_df = pd.DataFrame(bert_data)
    
    for col in ['Precision', 'Recall', 'F1']:
        bert_df[col] = pd.to_numeric(bert_df[col], errors='coerce')
    
    bert_df = bert_df.dropna(subset=['Precision', 'Recall', 'F1'])
    
    return bert_df, not bert_df.empty

def prepare_rouge_score_data(insights_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare ROUGE score data for visualization.
    """
    has_rouge_data = 'rouge_score' in insights_df.columns and not insights_df['rouge_score'].isna().all()
    
    if not has_rouge_data:
        return pd.DataFrame(), False
    
    rouge_data = []
    for _, row in insights_df.iterrows():
        if isinstance(row['rouge_score'], dict):
            query = row.get('question', '') or row.get('query', 'Unknown Query')
            rouge_data.append({
                'Query': query,
                'ROUGE-1': row['rouge_score'].get('rouge1_fmeasure', 0),
                'ROUGE-2': row['rouge_score'].get('rouge2_fmeasure', 0),
                'ROUGE-L': row['rouge_score'].get('rougeL_fmeasure', 0),
                'Is Correct': 'Correct' if row.get('is_correct', False) else 'Incorrect'
            })
    
    rouge_df = pd.DataFrame(rouge_data)
    
    for col in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        rouge_df[col] = pd.to_numeric(rouge_df[col], errors='coerce')
    
    rouge_df = rouge_df.dropna(subset=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    
    return rouge_df, not rouge_df.empty

def prepare_bert_score_by_groups_data(insights_df: pd.DataFrame, group_by: str = 'difficulty') -> pd.DataFrame:
    """
    Prepare BERT score data (Precision, Recall, F1) grouped by difficulty or tags.
    """
    if group_by not in ['difficulty', 'tags']:
        raise ValueError("group_by must be either 'difficulty' or 'tags'")
        
    data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            
            # Extract BERT scores
            bert_scores = record_dict.get('bert_score', {})
            if not isinstance(bert_scores, dict):
                continue
                
            scores = {
                'BERT Precision': bert_scores.get('bert_precision'),
                'BERT Recall': bert_scores.get('bert_recall'),
                'BERT F1': bert_scores.get('bert_f1')
            }
            
            if not all(isinstance(score, (int, float)) for score in scores.values()):
                continue
                
            # Handle the grouping based on the specified field
            if group_by == 'difficulty':
                group = record_dict.get('difficulty', 'Unknown')
                for score_type, score in scores.items():
                    data.append({
                        'Group': group,
                        'Score': score,
                        'Type': score_type
                    })
            else:  # tags
                tags = record_dict.get('tags', [])
                if isinstance(tags, list):
                    for tag in tags:
                        for score_type, score in scores.items():
                            data.append({
                                'Group': tag,
                                'Score': score,
                                'Type': score_type
                            })
                    continue
                else:
                    group = 'Untagged'
                    for score_type, score in scores.items():
                        data.append({
                            'Group': group,
                            'Score': score,
                            'Type': score_type
                        })
    
    # Convert to DataFrame and calculate statistics
    df = pd.DataFrame(data)
    if not df.empty:
        grouped = df.groupby(['Group', 'Type'])['Score'].agg(['mean', 'count']).reset_index()
        grouped.columns = ['Group', 'Type', 'Score', 'Count']
        return grouped
    return pd.DataFrame(columns=['Group', 'Type', 'Score', 'Count'])

def prepare_rouge_score_by_groups_data(insights_df: pd.DataFrame, group_by: str = 'difficulty') -> pd.DataFrame:
    """
    Prepare ROUGE score data grouped by difficulty or tags.
    """
    if group_by not in ['difficulty', 'tags']:
        raise ValueError("group_by must be either 'difficulty' or 'tags'")
        
    data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
            
            # Extract ROUGE scores
            rouge_scores = record_dict.get('rouge_score', {})
            if not isinstance(rouge_scores, dict):
                continue
                
            # Get F1 scores for each ROUGE metric
            rouge1_f1 = rouge_scores.get('rouge1_fmeasure')
            rouge2_f1 = rouge_scores.get('rouge2_fmeasure')
            rougeL_f1 = rouge_scores.get('rougeL_fmeasure')
            
            if not all(isinstance(score, (int, float)) for score in [rouge1_f1, rouge2_f1, rougeL_f1]):
                continue
                
            # Handle the grouping based on the specified field
            if group_by == 'difficulty':
                group = record_dict.get('difficulty', 'Unknown')
                for score, name in [(rouge1_f1, 'ROUGE-1'), (rouge2_f1, 'ROUGE-2'), (rougeL_f1, 'ROUGE-L')]:
                    data.append({
                        'Group': group,
                        'Score': score,
                        'Type': name
                    })
            else:  # tags
                tags = record_dict.get('tags', [])
                if isinstance(tags, list):
                    for tag in tags:
                        for score, name in [(rouge1_f1, 'ROUGE-1'), (rouge2_f1, 'ROUGE-2'), (rougeL_f1, 'ROUGE-L')]:
                            data.append({
                                'Group': tag,
                                'Score': score,
                                'Type': name
                            })
                    continue
                else:
                    group = 'Untagged'
                    for score, name in [(rouge1_f1, 'ROUGE-1'), (rouge2_f1, 'ROUGE-2'), (rougeL_f1, 'ROUGE-L')]:
                        data.append({
                            'Group': group,
                            'Score': score,
                            'Type': name
                        })
    
    # Convert to DataFrame and calculate statistics
    df = pd.DataFrame(data)
    if not df.empty:
        grouped = df.groupby(['Group', 'Type'])['Score'].agg(['mean', 'count']).reset_index()
        grouped.columns = ['Group', 'Type', 'Score', 'Count']
        return grouped
    return pd.DataFrame(columns=['Group', 'Type', 'Score', 'Count'])

def calculate_overall_metrics(insights_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall performance metrics for the RAG system.
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
    
    # Add context metrics from context_metrics module
    from .context_metrics import analyze_context_retrieval
    
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
