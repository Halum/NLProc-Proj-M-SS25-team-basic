"""
Utilities for retrieval analysis visualization
"""
import pandas as pd

def prepare_gold_context_presence_by_groups(insights_df: pd.DataFrame, group_option: str) -> pd.DataFrame:
    """
    Prepare data for gold context presence visualization grouped by selected option.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        group_option: Field to group by ('difficulty' or 'tags')
        
    Returns:
        DataFrame with gold context presence grouped by the selected option
    """
    # Ensure gold_context_pos is present
    if 'gold_context_pos' not in insights_df.columns:
        return pd.DataFrame()

    # Determine context presence based on gold_context_pos field
    insights_df['context_found'] = insights_df['gold_context_pos'] > 0

    # Validate group option
    if group_option not in ['difficulty', 'tags'] or group_option not in insights_df.columns:
        return pd.DataFrame()

    grouped_data = (
        insights_df.groupby(group_option)['context_found']
        .agg(['sum', 'size'])
        .reset_index()
        .rename(columns={'sum': 'Found', 'size': 'Total'})
    )
    
    grouped_data['Not Found'] = grouped_data['Total'] - grouped_data['Found']
    return grouped_data

def prepare_position_distribution_by_groups(insights_df: pd.DataFrame, group_option: str) -> pd.DataFrame:
    """
    Prepare data for context position distribution visualization grouped by selected option.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        group_option: Field to group by ('difficulty' or 'tags')
        
    Returns:
        DataFrame with position distribution grouped by the selected option
    """
    # Validate group option and check required columns
    if (group_option not in ['difficulty', 'tags'] or 
        group_option not in insights_df.columns or 
        'gold_context_pos' not in insights_df.columns):
        return pd.DataFrame()

    found_contexts = insights_df[insights_df['gold_context_pos'] > 0]
    if found_contexts.empty:
        return pd.DataFrame()

    # Group by both the selected option and position
    position_distribution = (
        found_contexts.groupby([group_option, 'gold_context_pos'])
        .size()
        .reset_index(name='count')
    )
    
    return position_distribution

def prepare_presence_by_correctness_groups(insights_df: pd.DataFrame, group_option: str) -> pd.DataFrame:
    """
    Prepare data for gold context presence by correctness visualization grouped by selected option.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        group_option: Field to group by ('difficulty' or 'tags')
        
    Returns:
        DataFrame with context presence by correctness grouped by the selected option
    """
    # Validate group option and ensure required columns exist
    if (group_option not in ['difficulty', 'tags'] or 
        not all(col in insights_df.columns for col in ['gold_context_pos', 'is_correct', group_option])):
        return pd.DataFrame()

    # Determine context presence based on gold_context_pos field
    insights_df['context_found'] = insights_df['gold_context_pos'] > 0

    # Group by both the selected option and correctness
    grouped_data = (
        insights_df.groupby([group_option, 'is_correct'])['context_found']
        .agg(['sum', 'size'])
        .reset_index()
        .rename(columns={'sum': 'Found', 'size': 'Total'})
    )
    
    # Calculate percentages
    grouped_data['Presence Rate'] = (grouped_data['Found'] / grouped_data['Total'] * 100).round(1)
    
    return grouped_data
