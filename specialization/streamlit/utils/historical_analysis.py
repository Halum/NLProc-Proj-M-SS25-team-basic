"""
Utilities for historical analysis data preparation.
"""
from typing import Dict, List, Tuple, Optional

import pandas as pd

def prepare_historical_bert_scores_by_groups(
    historical_data,
    group_by: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    """
    Prepare historical BERT scores data grouped by difficulty or tags.
    
    Args:
        historical_data: DataFrame containing historical metrics data
        group_by: Group by 'difficulty', 'tags', or None for overall scores
        
    Returns:
        Tuple containing:
        - Dictionary with groups and their BERT score trends
        - List of iteration labels
    """
    if group_by not in ['difficulty', 'tags', None]:
        raise ValueError("group_by must be either 'difficulty', 'tags', or None")
        
    result = {}
    iterations = []
    
    for idx, row in historical_data.iterrows():
        # Get iteration info
        iteration = f"Iter {idx + 1}"
        iterations.append(iteration)
        
        if group_by is None:
            # For overall scores, just use the averages
            result['all'] = result.get('all', {'precision': [], 'recall': [], 'f1': []})
            result['all']['precision'].append(row['avg_bert_precision'])
            result['all']['recall'].append(row['avg_bert_recall'])
            result['all']['f1'].append(row['avg_bert_f1'])
            continue
            
        # Get the insight data for this iteration
        insight_df = row['insight_data']
        if insight_df is None or group_by not in insight_df.columns:
            continue
            
        # Handle lists in tags column
        if group_by == 'tags':
            # Explode tags if they're in a list
            if isinstance(insight_df[group_by].iloc[0], list):
                insight_df = insight_df.explode(group_by)
                
        # Group by the specified column and calculate mean BERT scores
        grouped = insight_df.groupby(group_by).apply(
            lambda x: {
                'precision': x['bert_score'].apply(lambda s: s.get('bert_precision', 0)).mean(),
                'recall': x['bert_score'].apply(lambda s: s.get('bert_recall', 0)).mean(),
                'f1': x['bert_score'].apply(lambda s: s.get('bert_f1', 0)).mean(),
                'count': len(x)
            }
        ).to_dict()
        
        # Add to result dictionary
        for group, scores in grouped.items():
            if group not in result:
                result[group] = {'precision': [], 'recall': [], 'f1': []}
            result[group]['precision'].append(scores['precision'])
            result[group]['recall'].append(scores['recall'])
            result[group]['f1'].append(scores['f1'])
            
    return result, iterations

def prepare_historical_rouge_scores_by_groups(
    historical_data,
    group_by: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    """
    Prepare historical ROUGE scores data grouped by difficulty or tags.
    
    Args:
        historical_data: DataFrame containing historical metrics data
        group_by: Group by 'difficulty', 'tags', or None for overall scores
        
    Returns:
        Tuple containing:
        - Dictionary with groups and their ROUGE score trends
        - List of iteration labels
    """
    if group_by not in ['difficulty', 'tags', None]:
        raise ValueError("group_by must be either 'difficulty', 'tags', or None")
        
    result = {}
    iterations = []
    
    for idx, row in historical_data.iterrows():
        # Get iteration info
        iteration = f"Iter {idx + 1}"
        iterations.append(iteration)
        
        if group_by is None:
            # For overall scores, just use the averages
            result['all'] = result.get('all', {
                'rouge1': [], 'rouge2': [], 'rougeL': []
            })
            result['all']['rouge1'].append(row['avg_rouge1_f1'])
            result['all']['rouge2'].append(row['avg_rouge2_f1'])
            result['all']['rougeL'].append(row['avg_rougeL_f1'])
            continue
            
        # Get the insight data for this iteration
        insight_df = row['insight_data']
        if insight_df is None or group_by not in insight_df.columns:
            continue
            
        # Handle lists in tags column
        if group_by == 'tags':
            # Explode tags if they're in a list
            if isinstance(insight_df[group_by].iloc[0], list):
                insight_df = insight_df.explode(group_by)
                
        # Group by the specified column and calculate mean ROUGE scores
        grouped = insight_df.groupby(group_by).apply(
            lambda x: {
                'rouge1': x['rouge_score'].apply(lambda s: s.get('rouge1_fmeasure', 0)).mean(),
                'rouge2': x['rouge_score'].apply(lambda s: s.get('rouge2_fmeasure', 0)).mean(),
                'rougeL': x['rouge_score'].apply(lambda s: s.get('rougeL_fmeasure', 0)).mean(),
                'count': len(x)
            }
        ).to_dict()
        
        # Add to result dictionary
        for group, scores in grouped.items():
            if group not in result:
                result[group] = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            result[group]['rouge1'].append(scores['rouge1'])
            result[group]['rouge2'].append(scores['rouge2'])
            result[group]['rougeL'].append(scores['rougeL'])
            
    return result, iterations

def prepare_question_correctness_across_iterations(historical_data):
    """
    Prepare data for a stacked bar chart showing question correctness across iterations.
    
    Args:
        historical_data: DataFrame containing historical metrics with insight_data
        
    Returns:
        A dictionary with:
            - question_ids: List of all question IDs
            - iteration_data: List of dictionaries, each containing correct/incorrect counts by question_id
            - iterations: List of iteration labels
    """
    if historical_data.empty:
        return {'question_ids': [], 'iteration_data': [], 'iterations': []}
    
    # Create a set to collect all question IDs across all iterations
    all_question_ids = set()
    iteration_data = []
    iterations = []
    
    # Process each iteration (insight file)
    for idx, row in historical_data.iterrows():
        # Skip if no insight data available
        if 'insight_data' not in row or not isinstance(row['insight_data'], pd.DataFrame):
            continue
            
        insight_df = row['insight_data']
        
        # Skip if no id or is_correct columns
        if 'id' not in insight_df.columns or 'is_correct' not in insight_df.columns:
            continue
            
        # Create iteration label
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        iteration = f"Iter {idx + 1} ({correct}/{total})"
        iterations.append(iteration)
        
        # Get the correctness by question_id for this iteration
        question_data = {}
        for _, record in insight_df.iterrows():
            # Store the ID as both string (for dict keys) and integer (for sorting)
            try:
                numeric_id = int(record['id'])
                q_id = str(numeric_id)  # Store string version for dictionary keys
            except (ValueError, TypeError):
                q_id = str(record['id'])  # Fallback to string if conversion fails
                
            is_correct = bool(record['is_correct'])
            
            # Add to the set of all question IDs
            all_question_ids.add(q_id)
            
            # Store the correctness status
            question_data[q_id] = is_correct
            
        iteration_data.append(question_data)
    
    # Convert all question IDs to integers for proper numeric sorting
    numeric_ids = []
    for q_id in all_question_ids:
        try:
            numeric_ids.append(int(q_id))
        except (ValueError, TypeError):
            # Skip non-integer IDs for this chart
            pass
    
    # Sort numerically
    numeric_ids.sort()
    
    # Convert back to strings for consistency with the rest of the code
    question_ids = [str(num_id) for num_id in numeric_ids]
    
    return {
        'question_ids': question_ids,
        'iteration_data': iteration_data,
        'iterations': iterations
    }
