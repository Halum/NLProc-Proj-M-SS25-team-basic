"""
Insights tab view for visualizing chunking strategy insights
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def render_insights_ui():
    """
    Render the insights tab UI with visualizations
    
    Returns:
        insights_df: The loaded insights dataframe or None if not available
    """
    st.header("Chunking Strategy Insights")
    st.write("Visualize and compare insights from different chunking strategies.")
    
    # Load insights data
    insights_df = load_insights_data()
    
    # Display insights if available
    if insights_df is not None:
        display_insights(insights_df)
    else:
        st.warning("No insights data available. Process documents in the Preprocessing tab first.")
    
    return insights_df


def load_insights_data():
    """Load insights data for visualization"""
    insights_df = None
    try:
        # Get the directory where insight file should be stored
        insight_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "baseline", "insight")
        
        # Define possible insight file paths
        insight_path = os.path.join(insight_dir, "chunking_strategy_insights.csv")
        alternative_path = os.path.join(insight_dir, "chunking_strategy_insights.csv.csv")
        
        # Try to load the insights file
        if os.path.exists(insight_path):
            insights_df = pd.read_csv(insight_path)
            st.session_state.insights_loaded = True
        elif os.path.exists(alternative_path):
            insights_df = pd.read_csv(alternative_path)
            st.session_state.insights_loaded = True
    except Exception as e:
        st.warning(f"Could not load insights file: {str(e)}")
        st.session_state.insights_loaded = False
    
    return insights_df


def display_insights(insights_df):
    """Display strategy insights visualizations"""
    if insights_df is not None:
        # Filter insights by strategy if needed
        filtered_df = insights_df
        
        # Create visual insights with smaller graphs side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Number of chunks by strategy
            st.markdown("### Number of Chunks")
            chunks_by_strategy = filtered_df.groupby('chunk_strategy')['number_of_chunks'].mean().reset_index()
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.barplot(data=chunks_by_strategy, x='chunk_strategy', y='number_of_chunks', ax=ax1)
            ax1.set_xlabel('Strategy')
            ax1.set_ylabel('Number of Chunks')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # 2. Correct Answer Rate by Strategy
            st.markdown("### Correct Answer Rate")
            correct_answer_rate = filtered_df.groupby('chunk_strategy')['correct_answer'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=correct_answer_rate, x='chunk_strategy', y='correct_answer', ax=ax2)
            
            # Add percentage labels
            for i, bar in enumerate(bars.patches):
                bars.text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + 0.01, 
                        f'{bar.get_height():.0%}',
                        ha='center', va='bottom')
                        
            ax2.set_xlabel('Strategy')
            ax2.set_ylabel('Correct Answer Rate')
            ax2.set_ylim(0, 1.1)  # Set y-axis limit to accommodate percentage labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 3. Context Found Rate by Strategy
            st.markdown("### Context Found Rate")
            context_found_df = filtered_df.copy()
            context_found_df['context_found'] = context_found_df['retrieved_chunk_rank'] != -1
            context_found_rate = context_found_df.groupby('chunk_strategy')['context_found'].mean().reset_index()
            
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=context_found_rate, x='chunk_strategy', y='context_found', ax=ax3)
            
            # Add percentage labels
            for i, bar in enumerate(bars.patches):
                bars.text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + 0.01, 
                        f'{bar.get_height():.0%}',
                        ha='center', va='bottom')
                        
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('Context Found Rate')
            ax3.set_ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3)
        
        # Show raw data in the other column
        with col4:
            st.markdown("### Raw Insights Data")
            st.dataframe(filtered_df, height=300)
