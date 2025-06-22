"""
Query Explorer Page.

This page provides a detailed examination of individual queries
and their performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Add the project root to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import utilities for the dashboard
from specialization.streamlit.utils.data_loader import load_insight_data


st.set_page_config(
    page_title="Query Explorer",
    page_icon="🔍",
    layout="wide"
)

def display_query_details(query_data):
    """Display detailed information about a specific query"""
    st.header(f"Query: {query_data['question']}")
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Query Information")
        st.markdown(f"**Parsed Query:** {query_data['parsed_query']}")
        
        # Display metadata filters if available
        if query_data['metadata_filters'] and len(query_data['metadata_filters']) > 0:
            st.markdown("**Metadata Filters:**")
            st.json(query_data['metadata_filters'])
        else:
            st.markdown("**No metadata filters applied**")
        
        # Display gold answer
        st.markdown("**Gold Standard Answer:**")
        st.success(query_data['gold_answer'])
        
        # Display similarity score
        st.metric("Average Similarity Score", f"{query_data['avg_similarity_score']:.4f}")
    
    with col2:
        st.subheader("Generated Answer")
        
        # Display answer with correct/incorrect indicator
        if query_data['is_correct']:
            st.success(query_data['generated_answer'])
            st.markdown("✅ **Answer matches gold standard**")
        else:
            st.error(query_data['generated_answer'])
            st.markdown("❌ **Answer does not match gold standard**")
        
        # Display BERT scores if available
        if isinstance(query_data.get('bert_score'), dict):
            st.subheader("BERT Scores")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Precision", f"{query_data['bert_score'].get('bert_precision', 0):.4f}")
            
            with metrics_col2:
                st.metric("Recall", f"{query_data['bert_score'].get('bert_recall', 0):.4f}")
            
            with metrics_col3:
                st.metric("F1", f"{query_data['bert_score'].get('bert_f1', 0):.4f}")
        
        # Display ROUGE scores if available
        if isinstance(query_data.get('rouge_score'), dict):
            st.subheader("ROUGE Scores")
            rouge_cols = st.columns(3)
            
            with rouge_cols[0]:
                st.metric("ROUGE-1 F1", f"{query_data['rouge_score'].get('rouge1_fmeasure', 0):.4f}")
            
            with rouge_cols[1]:
                st.metric("ROUGE-2 F1", f"{query_data['rouge_score'].get('rouge2_fmeasure', 0):.4f}")
            
            with rouge_cols[2]:
                st.metric("ROUGE-L F1", f"{query_data['rouge_score'].get('rougeL_fmeasure', 0):.4f}")
    
    # Display context documents
    st.subheader("Retrieved Context Documents")
    
    if not query_data['context'] or len(query_data['context']) == 0:
        st.warning("No context documents retrieved.")
        return
    
    # Create a dataframe with context scores
    context_df = pd.DataFrame([
        {
            'Document Index': i + 1,
            'Content': ctx.get('content', ''),
            'Title': ctx.get('metadata', {}).get('title', 'Unknown'),
            'Similarity Score': ctx.get('score', 0),
            'Release Date': ctx.get('metadata', {}).get('release_date', 'Unknown'),
            'Revenue': ctx.get('metadata', {}).get('revenue', 0),
            'Vote Average': ctx.get('metadata', {}).get('vote_average', 0),
            'Runtime': ctx.get('metadata', {}).get('runtime', 0)
        }
        for i, ctx in enumerate(query_data['context'])
        if isinstance(ctx, dict)
    ])
    
    if not context_df.empty:
        # Plot context scores
        fig = px.bar(
            context_df,
            x='Document Index',
            y='Similarity Score',
            color='Similarity Score',
            color_continuous_scale='Viridis',
            labels={'Document Index': 'Document #', 'Similarity Score': 'Similarity'},
            title='Document Relevance Scores'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display context documents in expandable sections
        for i, ctx in context_df.iterrows():
            with st.expander(f"Document {ctx['Document Index']}: {ctx['Title']} (Score: {ctx['Similarity Score']:.4f})"):
                st.text(ctx['Content'])
                
                # Display metadata in columns
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                
                with meta_col1:
                    st.metric("Release Date", ctx['Release Date'])
                
                with meta_col2:
                    st.metric("Runtime (min)", ctx['Runtime'])
                
                with meta_col3:
                    st.metric("Revenue", f"${ctx['Revenue']:,.0f}")
                
                with meta_col4:
                    st.metric("Vote Average", ctx['Vote Average'])

def main():
    st.title("🔍 Query Explorer")
    
    # Load evaluation insights
    with st.spinner("Loading evaluation data..."):
        insights_df = load_insight_data()
        
    if insights_df is not None and not insights_df.empty:
        st.success(f"Loaded {len(insights_df)} evaluation records")
        
        # Create filter options
        st.sidebar.header("Filter Options")
        
        # Filter by correctness
        correctness_filter = st.sidebar.radio(
            "Answer Correctness:",
            options=["All", "Correct Only", "Incorrect Only"]
        )
        
        filtered_df = insights_df
        if correctness_filter == "Correct Only":
            filtered_df = insights_df[insights_df['is_correct']]
        elif correctness_filter == "Incorrect Only":
            filtered_df = insights_df[~insights_df['is_correct']]
        
        # Allow selection from queries
        query_list = filtered_df['question'].tolist()
        selected_query = st.sidebar.selectbox(
            "Select a query to explore:",
            options=query_list
        )
        
        # Display the selected query details
        if selected_query:
            query_data = filtered_df[filtered_df['question'] == selected_query].iloc[0].to_dict()
            display_query_details(query_data)
    else:
        st.error("Failed to load evaluation insights data. Please check the data file.")

if __name__ == "__main__":
    main()
