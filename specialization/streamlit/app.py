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
    plot_bert_scores,
    plot_rouge_scores
)

# App configuration
st.set_page_config(
    page_title="RAG Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)            # Custom CSS for consistent styling across the app
st.markdown("""
<style>
    /* Reduce sidebar width */
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 200px !important;
    }
    /* Add extra spacing between all horizontal blocks in the app */
    [data-testid="stHorizontalBlock"] {
        gap: 3rem !important;
    }
    /* Fix sidebar navigation text to be title case */
    section[data-testid="stSidebarUserContent"] .css-17lntkn {
        text-transform: capitalize !important;
    }
    section[data-testid="stSidebarUserContent"] .css-17lntkn:first-letter {
        text-transform: uppercase !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 RAG Performance Metrics Dashboard")
    
    # Get available insight files (only timestamped ones)
    from specialization.streamlit.utils.data_loader import get_available_insight_files
    available_files = get_available_insight_files()
    
    if not available_files:
        st.error("No timestamped insight files found. Please run the evaluation pipeline first.")
        st.info("Note: Only insight files with timestamps in their filenames are displayed. Recent evaluation runs automatically create timestamped files.")
        return
    
    # Create a dropdown for file selection
    file_options = [(path, f"Evaluation from {display_name}") for path, display_name, _ in available_files]
    selected_option = st.selectbox(
        "Select insight file:", 
        options=file_options,
        format_func=lambda x: x[1]
    )
    
    selected_file_path, selected_label = selected_option
    
    # Load the selected evaluation insights
    with st.spinner(f"Loading {selected_label}..."):
        insights_df = load_insight_data(selected_file_path)
        
    if insights_df is not None and not insights_df.empty:
        # Create a container with a border for the file info
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.success(f"Loaded {len(insights_df)} evaluation records from {selected_label}")
            
            with col2:
                if "timestamp" in insights_df.columns and not insights_df["timestamp"].empty:
                    # Get the earliest timestamp from the dataset
                    earliest_time = insights_df["timestamp"].min()
                    st.info(f"⏱️ Run date: {earliest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display overall metrics in the top section
        display_overall_metrics(insights_df)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "📈 Answer Performance", 
            "🔍 Retrieval Analysis", 
            "📋 Detailed Records"
        ])
        
        with tab1:
            # Add custom CSS for increased spacing between columns before creating the columns
            st.markdown("""
            <style>
            [data-testid="stHorizontalBlock"] > div:first-child {
                margin-right: 4rem;
                padding-right: 2rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.subheader("Answer Correctness")
            plot_answer_correctness(insights_df)
            
            st.markdown("---")
            st.subheader("Score Analysis")
            plot_bert_scores(insights_df)
            st.markdown("---")
            plot_rouge_scores(insights_df)
                
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
            records_per_page = 15
            total_pages = (len(filtered_df) + records_per_page - 1) // records_per_page
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
            
            start_idx = (page - 1) * records_per_page
            end_idx = min(start_idx + records_per_page, len(filtered_df))
            
            for i in range(start_idx, end_idx):
                record = filtered_df.iloc[i]
                with st.expander(f"**Query: {record['question']}**"):
                    col1, col2 = st.columns([1, 1], gap="large")
                    
                    with col1:
                        st.markdown("**Parsed Question:**")
                        st.info(record['parsed_question'])
                        
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
                        
                    # Create a new section for evaluation metrics
                    st.markdown("---")
                    st.markdown("### Evaluation Metrics")
                    metric_col1, metric_col2 = st.columns([1, 1], gap="large")
                    
                    with metric_col1:
                        # Display BERT scores if available
                        st.markdown("**BERT Scores:**")
                        if 'bert_score' in record and isinstance(record['bert_score'], dict):
                            bert_col1, bert_col2, bert_col3 = st.columns(3)
                            with bert_col1:
                                st.metric("Precision", f"{record['bert_score'].get('bert_precision', 'N/A'):.4f}" 
                                          if isinstance(record['bert_score'].get('bert_precision'), (int, float)) else "N/A")
                            with bert_col2:
                                st.metric("Recall", f"{record['bert_score'].get('bert_recall', 'N/A'):.4f}"
                                          if isinstance(record['bert_score'].get('bert_recall'), (int, float)) else "N/A")
                            with bert_col3:
                                st.metric("F1", f"{record['bert_score'].get('bert_f1', 'N/A'):.4f}"
                                          if isinstance(record['bert_score'].get('bert_f1'), (int, float)) else "N/A")
                        else:
                            st.info("BERT scores not available for this record")
                    
                    with metric_col2:
                        # Display ROUGE scores if available
                        st.markdown("**ROUGE Scores:**")
                        if 'rouge_score' in record and isinstance(record['rouge_score'], dict):
                            rouge_col1, rouge_col2, rouge_col3 = st.columns(3)
                            with rouge_col1:
                                st.metric("ROUGE-1 F1", f"{record['rouge_score'].get('rouge1_fmeasure', 'N/A'):.4f}"
                                          if isinstance(record['rouge_score'].get('rouge1_fmeasure'), (int, float)) else "N/A")
                            with rouge_col2:
                                st.metric("ROUGE-2 F1", f"{record['rouge_score'].get('rouge2_fmeasure', 'N/A'):.4f}"
                                          if isinstance(record['rouge_score'].get('rouge2_fmeasure'), (int, float)) else "N/A")
                            with rouge_col3:
                                st.metric("ROUGE-L F1", f"{record['rouge_score'].get('rougeL_fmeasure', 'N/A'):.4f}"
                                          if isinstance(record['rouge_score'].get('rougeL_fmeasure'), (int, float)) else "N/A")
                        else:
                            st.info("ROUGE scores not available for this record")
                    
                    st.markdown("---")
                    
                    # Display gold context first
                    st.markdown("### Gold Context")
                    if 'gold_context' in record and record['gold_context']:
                        if isinstance(record['gold_context'], list):
                            for j, ctx in enumerate(record['gold_context']):
                                if isinstance(ctx, dict):
                                    # Handle gold context that's a list of context dictionaries
                                    title = ctx.get('metadata', {}).get('title', f'Document {j+1}')
                                    st.markdown(f"**{title}**")
                                    st.text(ctx.get('content', 'No content available'))
                                else:
                                    # Handle gold context that's a list of strings
                                    st.markdown(f"**Document {j+1}**")
                                    st.text(ctx)
                        elif isinstance(record['gold_context'], dict):
                            # Handle gold context as a single dictionary
                            title = record['gold_context'].get('metadata', {}).get('title', 'Gold Document')
                            st.markdown(f"**{title}**")
                            st.text(record['gold_context'].get('content', 'No content available'))
                        else:
                            # Handle gold context as a string
                            st.text(str(record['gold_context']))
                    else:
                        st.info("No gold context available for this record")
                    
                    st.markdown("---")
                    
                    # Display retrieved context
                    st.markdown("### Retrieved Context")
                    for j, ctx in enumerate(record['context']):
                        st.markdown(f"**Document {j+1}** (Score: {ctx['score']:.4f}) - {ctx['metadata']['title']}")
                        st.text(ctx['content'])
    else:
        st.error("Failed to load evaluation insights data. Please check the data file.")

if __name__ == "__main__":
    main()
