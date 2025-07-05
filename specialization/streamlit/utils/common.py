"""
Common utility functions for data transformation and visualization.
"""

def prepare_common_grouping(df, group_option):
    """
    Common function to handle grouping by difficulty or tags.
    
    Args:
        df: DataFrame containing the data
        group_option: 'Difficulty' or 'Tags'
        
    Returns:
        grouped_df: DataFrame with proper grouping applied
        group_by: The column name to group by
    """
    group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    # For tags, we need to explode the tags column as it contains lists
    if group_by == 'tags':
        # Check if tags column exists and contains lists
        if 'tags' in df.columns and df['tags'].apply(lambda x: isinstance(x, list)).any():
            # Explode tags and reset index to maintain proper grouping
            df = df.explode('tags').reset_index(drop=True)
            # Fill any empty tags
            df['tags'] = df['tags'].fillna('No tags')
    
    return df, group_by
