"""Gold context analysis utilities for RAG evaluation insights."""
import pandas as pd


def prepare_gold_context_presence_data(insights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for gold context presence visualization using gold_context_pos field.
    
    Args:
        insights_df: DataFrame containing evaluation insights
        
    Returns:
        DataFrame with gold context retrieval positions and correctness information
    """
    # Process each record to collect gold context position and correctness info
    context_data = []
    
    for _, record in insights_df.iterrows():
        if isinstance(record, pd.Series):
            record = record.to_dict()
            
        # Get the position from gold_context_pos field 
        position = record.get('gold_context_pos', -1)
        if position == -1:
            position = None

        context_data.append({
            'Query': record.get('question', 'Unknown query'),
            'Position': position,
            'Is Correct': 'Correct' if record.get('is_correct', False) else 'Incorrect'
        })
    
    context_df = pd.DataFrame(context_data)
    return context_df
