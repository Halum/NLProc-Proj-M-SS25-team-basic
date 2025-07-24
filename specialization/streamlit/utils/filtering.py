"""
Filtering utilities for RAG Performance Dashboard.
Contains functions to filter and process insights data based on various criteria.
"""

import pandas as pd
from typing import List, Optional, Dict, Any

def filter_insights_data(
    insights_df: pd.DataFrame,
    correctness_filter: str = 'All',
    difficulty_levels: Optional[List[str]] = None,
    selected_tags: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter insights DataFrame based on multiple criteria.
    
    Args:
        insights_df (pd.DataFrame): Original insights DataFrame
        correctness_filter (str): Filter by correctness ('All', 'Correct Only', 'Incorrect Only')
        difficulty_levels (List[str], optional): Selected difficulty levels to filter by
        selected_tags (List[str], optional): Selected tags to filter by
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = insights_df.copy()
    
    # Apply correctness filter
    if correctness_filter == 'Correct Only':
        filtered_df = filtered_df[filtered_df['is_correct']]
    elif correctness_filter == 'Incorrect Only':
        filtered_df = filtered_df[~filtered_df['is_correct']]
    
    # Apply difficulty filter if selected
    if difficulty_levels:
        filtered_df = filtered_df[filtered_df['difficulty'].isin(difficulty_levels)]
    
    # Apply tags filter if selected
    if selected_tags:
        # Filter records that have any of the selected tags
        filtered_df = filtered_df[
            filtered_df['tags'].apply(lambda x: isinstance(x, list) and any(tag in x for tag in selected_tags))
        ]
    
    return filtered_df

def get_available_filters(insights_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract available filter options from the insights DataFrame.
    
    Args:
        insights_df (pd.DataFrame): Insights DataFrame
    
    Returns:
        Dict[str, List[str]]: Dictionary containing available filter options
    """
    filters = {
        'correctness': ['All', 'Correct Only', 'Incorrect Only'],
        'difficulty': [],
        'tags': []
    }
    
    # Get unique difficulty levels if the column exists
    if 'difficulty' in insights_df.columns:
        filters['difficulty'] = sorted(insights_df['difficulty'].dropna().unique().tolist())
    
    # Get unique tags if the column exists
    if 'tags' in insights_df.columns:
        # Flatten list of tags and get unique values
        all_tags = []
        for tags in insights_df['tags'].dropna():
            if isinstance(tags, list):
                all_tags.extend(tags)
        filters['tags'] = sorted(list(set(all_tags)))
    
    return filters

def style_record_header(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a styled header information for a record based on its correctness.
    
    Args:
        record (Dict[str, Any]): Record dictionary containing question and is_correct
    
    Returns:
        Dict[str, Any]: Dictionary containing header information
    """
    return {
        'text': f"Query: {record['question']}",
        'is_correct': record['is_correct']
    }
