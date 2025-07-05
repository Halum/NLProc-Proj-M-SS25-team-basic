"""
Context-related metrics and analysis for RAG evaluation.

This module contains functions for analyzing and processing context retrieval
metrics in the RAG system evaluation.
"""
from typing import Dict, Any
import pandas as pd

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
        
    for i, ctx in enumerate(retrieved_contexts[:5], 1):  # Check only top 5 contexts, 1-indexed position
        if isinstance(ctx, dict) and 'content' in ctx:
            ctx_content = ctx['content']
        elif isinstance(ctx, str):
            ctx_content = ctx
        else:
            continue
            
        # Normalize retrieved context
        ctx_content_norm = ctx_content.strip().lower()
        
        # Check if the normalized gold context is contained in this context
        if gold_content_norm in ctx_content_norm:
            result['position'] = i
            break
    
    return result

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
            analysis = analyze_context_retrieval(record.to_dict())
        elif isinstance(record, dict):
            analysis = analyze_context_retrieval(record)
        else:
            continue
        
        # Include ALL records, not just those where gold context was found
        context_data.append({
            'Query': analysis['query'],
            'Position': analysis['position'],
            'Is Correct': 'Correct' if analysis['is_correct'] else 'Incorrect'
        })
    
    context_df = pd.DataFrame(context_data)
    return context_df
