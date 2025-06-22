"""
RAG Performance Metrics Dashboard

This Streamlit application visualizes and analyzes performance metrics for a RAG 
(Retrieval Augmented Generation) system on a movie dataset.

The app provides visualizations and insights on:
1. Retrieval performance metrics
2. Answer accuracy and quality metrics
3. Detailed analysis of query parsing and filter performance
4. Comparison of similarity scores across different types of queries
"""

import streamlit as st
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utilities for the dashboard
from specialization.streamlit.utils.data_loader import load_insight_data
from specialization.streamlit.components.metrics_display import display_overall_metrics
from specialization.streamlit.components.charts import (
    plot_answer_correctness, 
    plot_similarity_distributions,
    plot_bert_rouge_scores
)

# App configuration
st.set_page_config(
    page_title="RAG Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📊 RAG Performance Metrics Dashboard")
    
    # Load evaluation insights
    with st.spinner("Loading evaluation data..."):
        insights_df = load_insight_data()
        
    if insights_df is not None and not insights_df.empty:
        st.success(f"Loaded {len(insights_df)} evaluation records")
        
        # Display overall metrics in the top section
        display_overall_metrics(insights_df)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "📈 Answer Performance", 
            "🔍 Retrieval Analysis", 
            "📋 Detailed Records"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Answer Correctness")
                plot_answer_correctness(insights_df)
                
            with col2:
                st.subheader("BERT & ROUGE Score Analysis")
                plot_bert_rouge_scores(insights_df)
                
        with tab2:
            st.subheader("Similarity Score Distribution")
            plot_similarity_distributions(insights_df)
            
            st.subheader("Context Analysis")
            st.write("Analysis of retrieved context and its relevance to the query")
            # Additional retrieval analysis will be added here
            
        with tab3:
            st.subheader("Evaluation Records")
            
            # Let users filter by correctness
            filter_options = ['All', 'Correct Only', 'Incorrect Only']
            selected_filter = st.selectbox("Filter results:", filter_options)
            
            filtered_df = insights_df
            if selected_filter == 'Correct Only':
                filtered_df = insights_df[insights_df['is_correct']]
            elif selected_filter == 'Incorrect Only':
                filtered_df = insights_df[~insights_df['is_correct']]
                
            # Display paginated records
            records_per_page = 5
            total_pages = (len(filtered_df) + records_per_page - 1) // records_per_page
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
            
            start_idx = (page - 1) * records_per_page
            end_idx = min(start_idx + records_per_page, len(filtered_df))
            
            for i in range(start_idx, end_idx):
                record = filtered_df.iloc[i]
                with st.expander(f"Query: {record['question']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Parsed Query:**")
                        st.info(record['parsed_query'])
                        
                        st.markdown("**Metadata Filters:**")
                        st.json(record['metadata_filters'])
                        
                        st.markdown("**Gold Answer:**")
                        st.success(record['gold_answer'])
                        
                    with col2:
                        st.markdown("**Generated Answer:**")
                        if record['is_correct']:
                            st.success(record['generated_answer'])
                        else:
                            st.error(record['generated_answer'])
                        
                        st.markdown("**Similarity Score:**")
                        st.metric("Avg. Similarity", f"{record['avg_similarity_score']:.4f}")
                        
                    st.markdown("**Retrieved Context:**")
                    for j, ctx in enumerate(record['context']):
                        st.markdown(f"**Document {j+1}** (Score: {ctx['score']:.4f}) - {ctx['metadata']['title']}")
                        st.text(ctx['content'])
    else:
        st.error("Failed to load evaluation insights data. Please check the data file.")

if __name__ == "__main__":
    main()
