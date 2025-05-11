"""
Insights tab view for visualizing chunking strategy insights
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
        # Display chunking strategy comparison table
        display_chunking_comparison_table(insights_df)
        
        # Display visualizations
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
        
        # Define insight file path
        insight_path = os.path.join(insight_dir, "chunking_strategy_insights.csv")
        
        # Try to load the insights file
        if os.path.exists(insight_path):
            insights_df = pd.read_csv(insight_path)
            st.session_state.insights_loaded = True
            
        # Convert similarity scores from string to list
        if insights_df is not None and 'similarity_scores' in insights_df.columns:
            # Safer approach to convert string representations to actual lists
            def safe_convert(x):
                if not isinstance(x, str):
                    return x
                try:
                    # Clean up the string representation before conversion
                    clean_str = x.strip().replace('  ', ' ')
                    return np.array([float(val) for val in clean_str.strip('[]').split()])
                except Exception:
                    return np.array([0.0])
                    
            insights_df['similarity_scores'] = insights_df['similarity_scores'].apply(safe_convert)
                
    except Exception as e:
        st.warning(f"Could not load insights file: {str(e)}")
        st.session_state.insights_loaded = False
    
    return insights_df


def display_chunking_comparison_table(insights_df):
    """Display chunking strategy comparison table similar to the one in the homework README"""
    st.markdown("### Chunking Strategy Comparison Table")
    
    # Process the data to match the table format in the README
    if insights_df is not None:
        # Calculate metrics per chunking strategy
        strategy_stats = []
        
        # Group by chunking strategy
        for strategy, group in insights_df.groupby('chunk_strategy'):
            # Get the number of chunks (should be consistent per strategy)
            total_chunks = group['number_of_chunks'].iloc[0]
            
            # Count correct and incorrect answers
            correct_answers = group['correct_answer'].sum()
            total_answers = len(group)
            incorrect_answers = total_answers - correct_answers
            
            # Calculate accuracy
            accuracy = correct_answers / total_answers * 100
            
            # Calculate average similarity scores
            avg_sim_correct = np.nan
            avg_sim_incorrect = np.nan
            
            # Get first similarity score for each entry
            if 'similarity_scores' in group.columns:
                try:
                    # Extract first similarity score from each row's similarity_scores array
                    def get_first_score(scores):
                        if hasattr(scores, '__len__') and len(scores) > 0:
                            return float(scores[0])
                        return np.nan
                    
                    group['first_sim_score'] = group['similarity_scores'].apply(get_first_score)
                    
                    # Calculate average for correct and incorrect separately
                    correct_mask = group['correct_answer']
                    if correct_mask.any():
                        avg_sim_correct = group.loc[correct_mask, 'first_sim_score'].mean()
                    
                    incorrect_mask = ~group['correct_answer']
                    if incorrect_mask.any():
                        avg_sim_incorrect = group.loc[incorrect_mask, 'first_sim_score'].mean()
                except Exception as e:
                    st.error(f"Error processing similarity scores: {e}")
            
            strategy_stats.append({
                'Chunking Strategy': strategy,
                'Total Chunks': total_chunks,
                'Correct Answers': correct_answers,
                'Incorrect Answers': incorrect_answers,
                'Accuracy': f"{accuracy:.0f}%",
                'Avg Similarity Score (Correct)': f"{avg_sim_correct:.2f}" if not pd.isna(avg_sim_correct) else "N/A",
                'Avg Similarity Score (Incorrect)': f"{avg_sim_incorrect:.2f}" if not pd.isna(avg_sim_incorrect) else "N/A"
            })
        
        # Create a DataFrame from the statistics
        stats_df = pd.DataFrame(strategy_stats)
        
        # Display the comparison table
        st.table(stats_df)
        
        # Add explanatory note
        st.markdown("*Note: Average similarity scores are calculated from the first value in the similarity scores array for each entry.*")


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
