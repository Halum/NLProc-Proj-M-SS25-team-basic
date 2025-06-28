"""
Insights tab view for visualizing chunking strategy insights
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys

# Add the src directory to the path to be able to import from baseline
src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import configuration
from baseline.config.config import INSIGHT_FOLDER_PATH, LOG_FILE_NAME

# Set up path to import utils from project, not from streamlit package
import sys
from os.path import abspath, dirname, join
# Append the streamlit directory to the path
streamlit_dir = dirname(dirname(abspath(__file__)))
if streamlit_dir not in sys.path:
    sys.path.append(streamlit_dir)
    
# Import get_short_strategy_name function
from utils.chunking_strategies import get_short_strategy_name

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
        # Add a filter for chunking strategies
        if 'chunk_strategy' in insights_df.columns:
            available_strategies = insights_df['chunk_strategy'].unique().tolist()
            selected_strategies = st.multiselect(
                "Select chunking strategies to compare:",
                options=available_strategies,
                default=available_strategies
            )
            
            # Filter insights by selected strategies
            if selected_strategies:
                filtered_insights_df = insights_df[insights_df['chunk_strategy'].isin(selected_strategies)]
            else:
                filtered_insights_df = insights_df
                st.warning("No strategies selected. Showing all data.")
        else:
            filtered_insights_df = insights_df
            st.warning("No chunking strategy information available in the insights data.")
        
        # Add group filtering if available
        if 'group_id' in insights_df.columns:
            available_groups = insights_df['group_id'].unique().tolist()
            if len(available_groups) > 1:
                selected_groups = st.multiselect(
                    "Select teams to compare:",
                    options=available_groups,
                    default=available_groups
                )
                
                # Filter insights by selected groups
                if selected_groups:
                    filtered_insights_df = filtered_insights_df[filtered_insights_df['group_id'].isin(selected_groups)]
        
        # Display chunking strategy comparison table
        display_chunking_comparison_table(filtered_insights_df)
        
        # Display visualizations
        display_insights(filtered_insights_df)
    else:
        st.warning("No insights data available. Process documents in the Preprocessing tab first.")
    
    return insights_df


def load_insights_data():
    """Load insights data for visualization"""
    insights_df = None
    try:
        # Create a more flexible path structure using project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Get path to insights file using config values
        insight_path = os.path.join(project_root, INSIGHT_FOLDER_PATH, f"{LOG_FILE_NAME}.json")
        st.info(f"Loading insights from: {insight_path}")
        
        # Try to load the insights file
        if os.path.exists(insight_path):
            try:
                insights_df = pd.read_json(insight_path)
                st.session_state.insights_loaded = True
                st.success(f"Successfully loaded insights from {os.path.basename(insight_path)}")
            except Exception as load_err:
                st.error(f"Error loading JSON file: {load_err}")
        else:
            st.warning(f"Insights file not found at: {insight_path}")
            st.session_state.insights_loaded = False
            
        # Convert similarity scores from string to list
        if insights_df is not None and 'similarity_scores' in insights_df.columns:
            # Safer approach to convert string representations to actual lists
            def safe_convert(x):
                if not isinstance(x, str):
                    return x
                try:
                    # Clean up the string representation before conversion
                    clean_str = x.strip().replace('  ', ' ')
                    # Handle array-like format
                    if clean_str.startswith('[') and clean_str.endswith(']'):
                        # Remove outer brackets and split by commas, handling potentially quoted values
                        values_str = clean_str[1:-1].strip()
                        # Handle nested list format that might come from the insight generator
                        if values_str.startswith('[') and values_str.endswith(']'):
                            values_str = values_str[1:-1].strip()
                        # Split and convert to float
                        if ',' in values_str:
                            values = [float(val.strip(' "\'')) for val in values_str.split(',') if val.strip(' "\'')]
                        else:
                            values = [float(val.strip(' "\'')) for val in values_str.split() if val.strip(' "\'')]
                        return np.array(values)
                    else:
                        # Handle space-separated list
                        return np.array([float(val) for val in clean_str.split() if val.strip()])
                except Exception as e:
                    st.error(f"Error converting similarity scores: {e}")
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
            avg_sim_mean = np.nan
            
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
            
            # Get average similarity mean if available
            if 'similarity_mean' in group.columns:
                avg_sim_mean = group['similarity_mean'].mean()
            
            # Get short name for the strategy
            short_name = get_short_strategy_name(strategy)
            
            # Add a tooltip that shows the full name when hovering over the short name
            strategy_display = f"{short_name} ({strategy})"
            
            strategy_stats.append({
                'Chunking Strategy': strategy_display,
                'Total Chunks': total_chunks,
                'Correct Answers': correct_answers,
                'Incorrect Answers': incorrect_answers,
                'Accuracy': f"{accuracy:.0f}%",
                'Avg Similarity Score (Correct)': f"{avg_sim_correct:.2f}" if not pd.isna(avg_sim_correct) else "N/A",
                'Avg Similarity Score (Incorrect)': f"{avg_sim_incorrect:.2f}" if not pd.isna(avg_sim_incorrect) else "N/A",
                'Avg Similarity Mean': f"{avg_sim_mean:.2f}" if not pd.isna(avg_sim_mean) else "N/A"
            })
        
        # Create a DataFrame from the statistics
        stats_df = pd.DataFrame(strategy_stats)
        
        # Display the comparison table
        st.table(stats_df)
        
        # Add explanatory note for the table
        st.markdown("*Note: Higher values are better for Correct Answers, Accuracy, and Similarity Scores. For Total Chunks, the optimal value depends on the specific use case.*")


def display_insights(insights_df):
    """Display strategy insights visualizations"""
    if insights_df is not None:
        # Filter insights by strategy if needed
        filtered_df = insights_df.copy()
        
        # Create a mapping of original strategy names to short names
        if 'chunk_strategy' in filtered_df.columns:
            # Create a new column with short strategy names for display
            filtered_df['strategy_short'] = filtered_df['chunk_strategy'].apply(get_short_strategy_name)
            
            # Create a mapping dictionary for reference
            strategy_mapping = {original: get_short_strategy_name(original) 
                              for original in filtered_df['chunk_strategy'].unique()}
        
        # Create visual insights with smaller graphs side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Number of chunks by strategy
            st.markdown("### Number of Chunks")
            chunks_by_strategy = filtered_df.groupby('strategy_short')['number_of_chunks'].mean().reset_index()
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.barplot(data=chunks_by_strategy, x='strategy_short', y='number_of_chunks', ax=ax1)
            ax1.set_xlabel('Strategy')
            ax1.set_ylabel('Number of Chunks')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
            st.markdown("*Note: This shows the average chunk count per strategy. Fewer chunks can mean less processing overhead, but effectiveness depends on other metrics.*")
        
        with col2:
            # 2. Correct Answer Rate by Strategy
            st.markdown("### Correct Answer Rate")
            correct_answer_rate = filtered_df.groupby('strategy_short')['correct_answer'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=correct_answer_rate, x='strategy_short', y='correct_answer', ax=ax2)
            
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
            st.markdown("*Note: Higher is better - indicates a more effective chunking strategy for accurate answers.*")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 3. Context Found Rate by Strategy
            st.markdown("### Context Found Rate")
            context_found_df = filtered_df.copy()
            context_found_df['context_found'] = context_found_df['retrieved_chunk_rank'] != -1
            context_found_rate = context_found_df.groupby('strategy_short')['context_found'].mean().reset_index()
            
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=context_found_rate, x='strategy_short', y='context_found', ax=ax3)
            
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
            st.markdown("*Note: Higher is better - shows the percentage of queries where relevant context was found in the results.*")
        
        # Show similarity mean in the other column
        with col4:
            # 4. Average Similarity Score by Strategy
            st.markdown("### Average Similarity Score")
            if 'similarity_mean' in filtered_df.columns:
                # Group by chunking strategy and calculate mean of similarity_mean
                sim_mean_by_strategy = filtered_df.groupby('strategy_short')['similarity_mean'].mean().reset_index()
                
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                bars = sns.barplot(data=sim_mean_by_strategy, x='strategy_short', y='similarity_mean', ax=ax4)
                
                # Add value labels
                for i, bar in enumerate(bars.patches):
                    bars.text(bar.get_x() + bar.get_width()/2., 
                            bar.get_height() + 0.01, 
                            f'{bar.get_height():.3f}',
                            ha='center', va='bottom')
                            
                ax4.set_xlabel('Strategy')
                ax4.set_ylabel('Average Similarity Score')
                ax4.set_ylim(0, max(sim_mean_by_strategy['similarity_mean']) * 1.1)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig4)
                st.markdown("*Note: Higher is better - higher similarity scores indicate better semantic matching between queries and chunks.*")
            else:
                st.warning("Similarity mean data not available in the insights file.")
        
        # Add a new row for detailed data display
        st.markdown("### Raw Insights Data")
        st.dataframe(filtered_df, height=300)
        
        # Add a section for question analysis if we have question data
        if 'question' in filtered_df.columns:
            st.markdown("### Question Analysis")
            
            # Group by question and analyze performance across chunking strategies
            question_analysis = filtered_df.groupby(['question', 'strategy_short'])['correct_answer'].mean().reset_index()
            question_analysis = question_analysis.pivot(index='question', columns='strategy_short', values='correct_answer')
            
            # Create a heatmap of question correctness across strategies
            if not question_analysis.empty:
                st.markdown("#### Question Performance by Chunking Strategy")
                st.write("This heatmap shows which questions were answered correctly (1.0) or incorrectly (0.0) by each chunking strategy.")
                
                # Configure figure size based on number of questions
                fig_height = max(6, len(question_analysis) * 0.4)
                fig5, ax5 = plt.subplots(figsize=(10, fig_height))
                
                # Create heatmap
                sns.heatmap(question_analysis, cmap="RdYlGn", vmin=0, vmax=1, 
                            annot=True, fmt=".1f", ax=ax5, cbar_kws={'label': 'Correct Answer (1=Yes, 0=No)'})
                
                ax5.set_ylabel('Question')
                ax5.set_xlabel('Chunking Strategy')
                plt.tight_layout()
                st.pyplot(fig5)
                st.markdown("*Note: Greener (closer to 1.0) is better - indicates questions correctly answered by that strategy.*")
                
        
        # Add a section for retrieved chunk rank analysis
        if 'retrieved_chunk_rank' in filtered_df.columns:
            st.markdown("### Retrieved Chunk Rank Analysis")
            st.write("This section shows which position (rank) contained the correct context across different chunking strategies.")
            
            # Filter out entries where context was not found (-1 values)
            rank_df = filtered_df[filtered_df['retrieved_chunk_rank'] >= 0].copy()
            
            if not rank_df.empty:
                col_rank1, col_rank2 = st.columns(2)
                
                with col_rank1:
                    # 1. Average rank by strategy
                    st.markdown("#### Average Position of Correct Context")
                    avg_rank_by_strategy = rank_df.groupby('strategy_short')['retrieved_chunk_rank'].mean().reset_index()
                    avg_rank_by_strategy['retrieved_chunk_rank'] = avg_rank_by_strategy['retrieved_chunk_rank']
                    
                    fig_rank1, ax_rank1 = plt.subplots(figsize=(5, 4))
                    bars = sns.barplot(data=avg_rank_by_strategy, x='strategy_short', y='retrieved_chunk_rank', ax=ax_rank1)
                    
                    # Add value labels
                    for i, bar in enumerate(bars.patches):
                        bars.text(bar.get_x() + bar.get_width()/2., 
                                bar.get_height() + 0.05, 
                                f'{bar.get_height():.2f}',
                                ha='center', va='bottom')
                    
                    ax_rank1.set_xlabel('Strategy')
                    ax_rank1.set_ylabel('Average Position (lower is better)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_rank1)
                    
                    st.markdown("*Note: Lower values are better - they indicate the relevant context was found sooner in the results.*")
                
                with col_rank2:
                    # 2. Distribution of ranks (histogram)
                    st.markdown("#### Distribution of Context Positions")
                    # Add 1 to rank for display (1-based instead of 0-based)
                    rank_df['rank_display'] = rank_df['retrieved_chunk_rank']
                    
                    fig_rank2, ax_rank2 = plt.subplots(figsize=(5, 4))
                    # Count occurrences by rank and strategy
                    rank_counts = rank_df.groupby(['strategy_short', 'rank_display']).size().reset_index(name='count')
                    sns.barplot(data=rank_counts, x='rank_display', y='count', hue='strategy_short', ax=ax_rank2)
                    
                    ax_rank2.set_xlabel('Position of Correct Context')
                    ax_rank2.set_ylabel('Count')
                    ax_rank2.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig_rank2)
                    st.markdown("*Note: Higher counts at lower positions (left side of chart) are better - they indicate the relevant context was found earlier in the results.*")
                
                # 3. Table showing top-k recall rates (in what percentage of cases was the context found in the top k positions)
                st.markdown("#### Top-K Recall Rates")
                st.write("This table shows the percentage of queries where the context was found within the top K results.")
                
                # Calculate top-k recall rates (top-1, top-3, top-5)
                topk_stats = []
                # We need to map the strategy back to shortened version
                for strategy, group in rank_df.groupby('chunk_strategy'):
                    # Count total queries for this strategy (including those where context was not found)
                    total_queries = len(filtered_df[filtered_df['chunk_strategy'] == strategy])
                    
                    # Get short name for the strategy
                    short_name = get_short_strategy_name(strategy)
                    
                    # Calculate recall rates
                    top1_recall = (group['retrieved_chunk_rank'] <= 1).sum() / total_queries
                    top3_recall = (group['retrieved_chunk_rank'] <= 3).sum() / total_queries
                    top5_recall = (group['retrieved_chunk_rank'] <= 5).sum() / total_queries
                    
                    topk_stats.append({
                        'Chunking Strategy': f"{short_name} ({strategy})",
                        'Top-1 Recall': f"{top1_recall:.1%}",
                        'Top-3 Recall': f"{top3_recall:.1%}",
                        'Top-5 Recall': f"{top5_recall:.1%}"
                    })
                
                # Create DataFrame and display as table
                topk_df = pd.DataFrame(topk_stats)
                st.table(topk_df)
                st.markdown("*Note: Higher percentages are better - they indicate a greater proportion of queries where the correct context was found within the top K positions.*")
            else:
                st.warning("No valid retrieved chunk rank data available (all values are -1).")
        
        # Add a correlation analysis section
        st.markdown("### Correlation Analysis")
        st.write("This section explores correlations between correct answers, retrieved chunk position, and similarity scores.")
        
        # Only proceed if we have the necessary columns
        if all(col in filtered_df.columns for col in ['correct_answer', 'retrieved_chunk_rank']):
            # Create a new row for correlation visualizations
            corr_col1, corr_col2 = st.columns(2)
            
            # Only include rows where context was found
            corr_df = filtered_df[filtered_df['retrieved_chunk_rank'] >= 0].copy()
            
            with corr_col1:
                # Scatter plot: Retrieved Chunk Rank vs Correct Answer
                st.markdown("#### Retrieved Chunk Position vs Correct Answers")
                fig_corr1, ax_corr1 = plt.subplots(figsize=(6, 4))
                
                # Jitter the correct_answer values slightly for better visualization
                corr_df['jittered_correct'] = corr_df['correct_answer'] + np.random.normal(0, 0.05, len(corr_df))
                
                sns.scatterplot(
                    data=corr_df,
                    x='retrieved_chunk_rank',
                    y='jittered_correct',
                    hue='strategy_short',
                    alpha=0.7,
                    ax=ax_corr1
                )
                
                ax_corr1.set_xlabel('Retrieved Chunk Position')
                ax_corr1.set_ylabel('Correct Answer (1=Yes, 0=No)')
                ax_corr1.set_yticks([0, 1])
                ax_corr1.set_yticklabels(['No', 'Yes'])
                ax_corr1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig_corr1)
                
                # Add some statistics
                correct_by_rank = corr_df.groupby('retrieved_chunk_rank')['correct_answer'].mean()
                st.markdown(f"**Correlation coefficient:** {corr_df['retrieved_chunk_rank'].corr(corr_df['correct_answer']):.3f}")
                st.markdown("*Note: Lower chunk positions (chunks ranked higher) tend to yield more correct answers.*")
            
            with corr_col2:
                # If similarity_mean is available, create a box plot
                if 'similarity_mean' in corr_df.columns:
                    st.markdown("#### Similarity Scores vs Answer Correctness")
                    fig_corr2, ax_corr2 = plt.subplots(figsize=(6, 4))
                    
                    # Convert boolean to categorical for better labeling
                    corr_df['Answer'] = corr_df['correct_answer'].map({True: 'Correct', False: 'Incorrect'})
                    
                    sns.boxplot(
                        data=corr_df,
                        x='Answer',
                        y='similarity_mean',
                        ax=ax_corr2
                    )
                    
                    ax_corr2.set_xlabel('Answer Correctness')
                    ax_corr2.set_ylabel('Mean Similarity Score')
                    plt.tight_layout()
                    st.pyplot(fig_corr2)
                    
                    # Add some statistics
                    correct_sim = corr_df[corr_df['correct_answer']]['similarity_mean'].mean()
                    incorrect_sim = corr_df[~corr_df['correct_answer']]['similarity_mean'].mean()
                    st.markdown(f"**Avg similarity for correct answers:** {correct_sim:.3f}")
                    st.markdown(f"**Avg similarity for incorrect answers:** {incorrect_sim:.3f}")
                    st.markdown("*Note: Higher similarity scores tend to correlate with correct answers.*")
            
            # Add a heatmap showing correlations between metrics
            st.markdown("#### Correlation Heatmap")
            
            # Prepare data for correlation analysis
            corr_metrics = corr_df[['correct_answer', 'retrieved_chunk_rank', 'number_of_chunks']].copy()
            
            # Add similarity_mean if available
            if 'similarity_mean' in corr_df.columns:
                corr_metrics['similarity_mean'] = corr_df['similarity_mean']
            
            # Add min similarity score if similarity_scores is available
            if 'similarity_scores' in corr_df.columns:
                # Get minimum score from each row's similarity_scores array
                corr_metrics['min_similarity'] = corr_df['similarity_scores'].apply(
                    lambda x: min(x) if hasattr(x, '__len__') and len(x) > 0 else np.nan
                )
            
            # Rename columns for better display
            corr_metrics = corr_metrics.rename(columns={
                'correct_answer': 'Correct Answer',
                'retrieved_chunk_rank': 'Retrieved Position',
                'number_of_chunks': 'Chunk Count',
                'similarity_mean': 'Avg Similarity',
                'min_similarity': 'Min Similarity'
            })
            
            # Calculate correlation matrix
            corr_matrix = corr_metrics.corr()
            
            # Create correlation heatmap
            fig_corr3, ax_corr3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                ax=ax_corr3
            )
            plt.tight_layout()
            st.pyplot(fig_corr3)
            
            st.markdown("""
            *Note on interpreting correlations:*
            - Values close to 1 indicate strong positive correlation
            - Values close to -1 indicate strong negative correlation
            - Values close to 0 indicate little to no correlation
            - We expect negative correlation between Retrieved Position and Correct Answer (lower position = better)
            - We expect positive correlation between Similarity scores and Correct Answer
            
            **Important:** Negative correlations for similarity scores might indicate:
            1. An inverse relationship where lower similarity scores result in better answers (unusual)
            2. Data anomalies or outliers affecting the correlation calculation
            3. Inconsistencies in how similarity scores were calculated across different strategies
            4. Small sample sizes leading to statistically insignificant correlations
            
            If you see negative correlations between similarity scores and correct answers, you may want to examine the 
            raw data more closely to understand the underlying pattern.
            """)
            
            # Add detailed correlation breakdown
            st.markdown("#### Similarity Score Detailed Analysis")
            
            if 'similarity_scores' in corr_df.columns:
                # Create scatter plots of min/max similarity vs correctness
                corr_detail_col1, corr_detail_col2 = st.columns(2)
                
                with corr_detail_col1:
                    # Calculate first/top similarity score (usually most relevant)
                    corr_df['top_similarity'] = corr_df['similarity_scores'].apply(
                        lambda x: max(x) if hasattr(x, '__len__') and len(x) > 0 else np.nan
                    )
                    
                    # Create violin plot for top similarity score by answer correctness
                    st.markdown("##### Top Similarity Score Distribution")
                    fig_sim1, ax_sim1 = plt.subplots(figsize=(6, 4))
                    
                    # Create violin plot
                    sns.violinplot(
                        data=corr_df,
                        x='Answer',
                        y='top_similarity',
                        ax=ax_sim1
                    )
                    
                    ax_sim1.set_xlabel('Answer Correctness')
                    ax_sim1.set_ylabel('Top Similarity Score')
                    plt.tight_layout()
                    st.pyplot(fig_sim1)
                
                with corr_detail_col2:
                    # Calculate correlation values specifically for similarity metrics
                    sim_corr = pd.DataFrame()
                    if 'top_similarity' in corr_df.columns:
                        sim_corr['Top Similarity'] = [corr_df['correct_answer'].corr(corr_df['top_similarity'])]
                    if 'similarity_mean' in corr_df.columns:
                        sim_corr['Avg Similarity'] = [corr_df['correct_answer'].corr(corr_df['similarity_mean'])]
                    if 'min_similarity' in corr_df.columns:
                        sim_corr['Min Similarity'] = [corr_df['correct_answer'].corr(corr_df['min_similarity'])]
                    
                    sim_corr.index = ['Correlation with Correct Answer']
                    
                    # Display the correlation values
                    st.markdown("##### Similarity Correlations with Correct Answers")
                    st.table(sim_corr.T)
                    
                    st.markdown("""
                    **Interpreting these values:**
                    - Positive values indicate that higher similarity corresponds to more correct answers
                    - Negative values suggest that lower similarity corresponds to more correct answers
                    - Values close to zero indicate little relationship between similarity and correctness
                    
                    Unexpected correlations may result from:
                    - How chunks are selected and ranked
                    - Quality of embeddings for different types of questions
                    - Interactions between chunk size and content relevance
                    """)
        else:
            st.warning("Correlation analysis requires correct_answer and retrieved_chunk_rank columns.")
