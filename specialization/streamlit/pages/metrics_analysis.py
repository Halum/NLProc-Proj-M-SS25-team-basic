"""
Metric Detail Analysis Page.

This page provides in-depth analysis of RAG system performance metrics.
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
    page_title="Metric Analysis",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS for consistent styling
st.markdown("""
<style>
    /* Reduce sidebar width */
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 200px !important;
    }
    /* Add extra spacing between horizontal blocks */
    [data-testid="stHorizontalBlock"] {
        gap: 3rem !important;
    }
    [data-testid="stHorizontalBlock"] > div:first-child {
        margin-right: 4rem;
        padding-right: 2rem;
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

def analyze_bert_scores(insights_df):
    """Analyze BERT scores in detail"""
    st.header("BERT Score Analysis")
    
    # Extract BERT scores
    bert_scores = []
    for idx, row in insights_df.iterrows():
        if isinstance(row.get('bert_score'), dict):
            bert_scores.append({
                'Query': row['question'],
                'Precision': row['bert_score'].get('bert_precision', 0),
                'Recall': row['bert_score'].get('bert_recall', 0),
                'F1': row['bert_score'].get('bert_f1', 0),
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect',
                'Avg. Similarity': row['avg_similarity_score']
            })
    
    if not bert_scores:
        st.warning("No BERT score data available.")
        return
    
    bert_df = pd.DataFrame(bert_scores)
    
    # Display summary statistics
    st.subheader("BERT Score Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. BERT Precision", f"{bert_df['Precision'].mean():.4f}")
    
    with col2:
        st.metric("Avg. BERT Recall", f"{bert_df['Recall'].mean():.4f}")
    
    with col3:
        st.metric("Avg. BERT F1", f"{bert_df['F1'].mean():.4f}")
    
    # Create scatter plot of BERT F1 vs similarity score
    fig = px.scatter(
        bert_df,
        x='Avg. Similarity',
        y='F1',
        color='Is Correct',
        hover_data=['Query'],
        trendline="ols",
        color_discrete_map={'Correct': '#4CAF50', 'Incorrect': '#F44336'},
        title='BERT F1 Score vs. Average Similarity Score'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create correlation heatmap
    numeric_cols = ['Precision', 'Recall', 'F1', 'Avg. Similarity']
    corr_matrix = bert_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Correlation Between BERT Scores and Similarity'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def analyze_retrieval_performance(insights_df):
    """Analyze retrieval performance in detail"""
    st.header("Retrieval Performance Analysis")
    
    # Analyze similarity scores
    similarity_scores = insights_df['avg_similarity_score'].tolist()
    
    if not similarity_scores:
        st.warning("No similarity score data available.")
        return
    
    # Summary statistics
    st.subheader("Similarity Score Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Minimum", f"{min(similarity_scores):.4f}")
    
    with col2:
        st.metric("Maximum", f"{max(similarity_scores):.4f}")
    
    with col3:
        st.metric("Average", f"{np.mean(similarity_scores):.4f}")
    
    with col4:
        st.metric("Median", f"{np.median(similarity_scores):.4f}")
    
    # Create histogram of similarity scores
    fig = px.histogram(
        insights_df,
        x='avg_similarity_score',
        color='is_correct',
        marginal='box',
        nbins=20,
        histnorm='probability density',
        color_discrete_map={True: '#4CAF50', False: '#F44336'},
        title='Distribution of Similarity Scores'
    )
    
    fig.update_layout(
        xaxis_title='Average Similarity Score',
        yaxis_title='Density',
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze the number of context chunks that contained relevant information
    st.subheader("Context Analysis")
    st.write("This analysis shows how many context documents were retrieved and their relevance scores.")
    
    # Extract context information
    context_data = []
    for idx, row in insights_df.iterrows():
        if isinstance(row.get('context'), list):
            for i, ctx in enumerate(row['context']):
                if isinstance(ctx, dict) and 'score' in ctx:
                    context_data.append({
                        'Query': row['question'],
                        'Context Index': i + 1,
                        'Similarity Score': ctx['score'],
                        'Is Correct Answer': 'Yes' if row['is_correct'] else 'No'
                    })
    
    if context_data:
        context_df = pd.DataFrame(context_data)
        
        # Plot similarity scores by context position
        fig = px.box(
            context_df, 
            x='Context Index', 
            y='Similarity Score',
            color='Is Correct Answer',
            color_discrete_map={'Yes': '#4CAF50', 'No': '#F44336'},
            title='Context Similarity Scores by Position'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📈 RAG Metric Analysis")
    
    # Load evaluation insights
    with st.spinner("Loading evaluation data..."):
        insights_df = load_insight_data()
        
    if insights_df is not None and not insights_df.empty:
        st.success(f"Loaded {len(insights_df)} evaluation records")
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs([
            "BERT Score Analysis", 
            "Retrieval Analysis"
        ])
        
        with tab1:
            analyze_bert_scores(insights_df)
            
        with tab2:
            analyze_retrieval_performance(insights_df)
    else:
        st.error("Failed to load evaluation insights data. Please check the data file.")

if __name__ == "__main__":
    main()
