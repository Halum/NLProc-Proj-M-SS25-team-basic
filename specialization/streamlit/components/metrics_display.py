"""
Component for displaying overall RAG performance metrics.
"""

import streamlit as st

# Import data transformation utilities
from specialization.streamlit.utils import calculate_overall_metrics

def display_overall_metrics(insights_df):
    """
    Display overall performance metrics for the RAG system.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to calculate metrics
    metrics = calculate_overall_metrics(insights_df)
    
    # Extract metrics for display
    total_queries = metrics['total_queries']
    correct_answers = metrics['correct_answers']
    correct_percent = metrics['accuracy_percent']
    avg_similarity = metrics['avg_similarity']
    avg_bert_f1 = metrics.get('avg_bert_f1')
    avg_rouge_f1 = metrics.get('avg_rouge_f1')
    
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
