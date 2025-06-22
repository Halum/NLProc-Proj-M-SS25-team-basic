"""
Component for displaying overall RAG performance metrics.
"""

import streamlit as st
import numpy as np

def display_overall_metrics(insights_df):
    """
    Display overall performance metrics for the RAG system.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Calculate metrics
    total_queries = len(insights_df)
    correct_answers = insights_df['is_correct'].sum()
    correct_percent = (correct_answers / total_queries) * 100 if total_queries > 0 else 0
    
    avg_similarity = insights_df['avg_similarity_score'].mean()
    
    # Calculate average BERT score if available
    if 'bert_score' in insights_df.columns and not insights_df['bert_score'].isna().all():
        avg_bert_f1 = np.mean([score.get('bert_f1', 0) for score in insights_df['bert_score'] if isinstance(score, dict)])
    else:
        avg_bert_f1 = None
    
    # Calculate average ROUGE score if available
    if 'rouge_score' in insights_df.columns and not insights_df['rouge_score'].isna().all():
        avg_rouge_f1 = np.mean([score.get('rouge1_fmeasure', 0) for score in insights_df['rouge_score'] if isinstance(score, dict)])
    else:
        avg_rouge_f1 = None
    
    # Display metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy", 
            f"{correct_percent:.1f}%",
            f"{correct_answers}/{total_queries} queries"
        )
    
    with col2:
        st.metric(
            "Avg. Similarity Score",
            f"{avg_similarity:.4f}"
        )
    
    with col3:
        if avg_bert_f1 is not None:
            st.metric(
                "Avg. BERT F1 Score",
                f"{avg_bert_f1:.4f}"
            )
        else:
            st.metric(
                "Avg. BERT F1 Score",
                "N/A"
            )
    
    with col4:
        if avg_rouge_f1 is not None:
            st.metric(
                "Avg. ROUGE F1 Score",
                f"{avg_rouge_f1:.4f}"
            )
        else:
            st.metric(
                "Avg. ROUGE F1 Score",
                "N/A"
            )
