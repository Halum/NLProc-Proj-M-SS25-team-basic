"""
Visualization components for RAG performance metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Import data transformation utilities
from specialization.streamlit.utils.data_transformation import (
    calculate_dynamic_chart_height, 
    prepare_correctness_data,
    prepare_similarity_distribution_data,
    prepare_bert_score_data,
    prepare_rouge_score_data,
    prepare_gold_context_presence_data
)

def plot_answer_correctness(insights_df):
    """
    Create a visualizations for answer correctness.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to prepare the data
    correct_counts = prepare_correctness_data(insights_df)
    
    # Extract data for the chart
    values = correct_counts['Count'].tolist()
    labels = correct_counts['Is Correct'].tolist()
    
    # Create color map making sure Incorrect is always red
    color_map = {
        'Correct': '#4CAF50',
        'Incorrect': '#F44336'
    }
    colors = [color_map[label] for label in labels]
    
    # Create pie chart manually with go.Pie
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title='Answer Correctness Distribution',
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def plot_similarity_distributions(insights_df):
    """
    Create visualizations for similarity score distributions.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to prepare the data
    plot_df = prepare_similarity_distribution_data(insights_df)
    
    # Split data by correctness
    correct_df = plot_df[plot_df['Is Correct'] == 'Correct']
    incorrect_df = plot_df[plot_df['Is Correct'] == 'Incorrect']
    
    # Define bin parameters for the histogram (0 to 1 with steps of 0.05)
    
    # Create figure manually with go.Histogram for more control
    fig = go.Figure()
    
    # Add histogram for correct answers
    if not correct_df.empty:
        fig.add_trace(go.Histogram(
            x=correct_df['Average Similarity'],
            name='Correct',
            marker_color='#4CAF50',
            xbins=dict(start=0, end=1, size=0.05),  # Explicit bin definition
            opacity=0.7,
            hovertemplate='Similarity: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))
    
    # Add histogram for incorrect answers
    if not incorrect_df.empty:
        fig.add_trace(go.Histogram(
            x=incorrect_df['Average Similarity'],
            name='Incorrect',
            marker_color='#F44336',
            xbins=dict(start=0, end=1, size=0.05),  # Explicit bin definition
            opacity=0.7,
            hovertemplate='Similarity: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Distribution of Similarity Scores by Answer Correctness',
        xaxis_title='Average Similarity Score',
        yaxis_title='Count',
        barmode='group',  # Group bars side by side
        bargap=0.1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Format x-axis to show decimals
    fig.update_xaxes(tickformat=".2f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate dynamic height based on number of entries and query text length
    chart_height = calculate_dynamic_chart_height(plot_df)
    
    # Sort by Average Similarity for better visualization
    plot_df = plot_df.sort_values(by='Average Similarity', ascending=True)
    
    # Create horizontal bar chart of similarity scores vs. query
    fig = go.Figure()
    
    # Split data by correctness to create separate bars
    correct_df = plot_df[plot_df['Is Correct'] == 'Correct']
    incorrect_df = plot_df[plot_df['Is Correct'] == 'Incorrect']
    
    # Add bars for correct answers
    if not correct_df.empty:
        fig.add_trace(
            go.Bar(
                y=correct_df['Query'].tolist(),
                x=correct_df['Average Similarity'].tolist(),
                name='Correct',
                marker_color='#4CAF50',
                orientation='h',
                text=[f"{x:.4f}" for x in correct_df['Average Similarity'].tolist()],  # Format to 4 decimal places
                textposition='auto'
            )
        )
    
    # Add bars for incorrect answers
    if not incorrect_df.empty:
        fig.add_trace(
            go.Bar(
                y=incorrect_df['Query'].tolist(),
                x=incorrect_df['Average Similarity'].tolist(),
                name='Incorrect',
                marker_color='#F44336',
                orientation='h',
                text=[f"{x:.4f}" for x in incorrect_df['Average Similarity'].tolist()],  # Format to 4 decimal places
                textposition='auto'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Similarity Scores by Query',
        height=chart_height,
        xaxis_title='Average Similarity Score',
        yaxis_title='Query',
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Ensure x-axis shows proper decimal formatting
    fig.update_xaxes(tickformat=".4f")
    
    st.plotly_chart(fig, use_container_width=True)
    
def plot_bert_scores(insights_df):
    """
    Create visualization for BERT scores.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to prepare the data
    bert_df, has_bert_data = prepare_bert_score_data(insights_df)
    
    if not has_bert_data:
        st.warning("No BERT scores available in this dataset.")
        return
    
    # Create figure for BERT scores
    fig = go.Figure()
    
    # Add BERT score traces
    fig.add_trace(
        go.Bar(
            y=bert_df['Query'].tolist(),
            x=bert_df['Precision'].tolist(),
            name='BERT Precision',
            marker_color='#1f77b4',
            orientation='h'
        )
    )
    
    fig.add_trace(
        go.Bar(
            y=bert_df['Query'].tolist(),
            x=bert_df['Recall'].tolist(),
            name='BERT Recall',
            marker_color='#ff7f0e',
            orientation='h'
        )
    )
    
    fig.add_trace(
        go.Bar(
            y=bert_df['Query'].tolist(),
            x=bert_df['F1'].tolist(),
            name='BERT F1',
            marker_color='#2ca02c',
            orientation='h'
        )
    )
    
    # Create a separate trace for correctness indicators
    correct_markers = []
    incorrect_markers = []
    
    for i, correct in enumerate(bert_df['Is Correct']):
        if correct == 'Correct':
            correct_markers.append(i)
        else:
            incorrect_markers.append(i)
            
    # Add markers for correct answers
    if correct_markers:
        fig.add_trace(
            go.Scatter(
                x=[-0.05] * len(correct_markers),
                y=[bert_df['Query'].tolist()[i] for i in correct_markers],
                mode='markers+text',
                marker=dict(symbol='circle', color='#4CAF50', size=10),
                text=['✓'] * len(correct_markers),
                textposition='middle center',
                textfont=dict(color='white'),
                name='Correct',
                hoverinfo='none'
            )
        )
        
    # Add markers for incorrect answers
    if incorrect_markers:
        fig.add_trace(
            go.Scatter(
                x=[-0.05] * len(incorrect_markers),
                y=[bert_df['Query'].tolist()[i] for i in incorrect_markers],
                mode='markers+text',
                marker=dict(symbol='circle', color='#F44336', size=10),
                text=['✗'] * len(incorrect_markers),
                textposition='middle center',
                textfont=dict(color='white'),
                name='Incorrect',
                hoverinfo='none'
            )
        )
    
    # Calculate dynamic height based on number of entries and query text length
    chart_height = calculate_dynamic_chart_height(bert_df)
    
    # Update layout
    fig.update_layout(
        title="BERT Scores <br>(Higher scores indicate better semantic matching)",
        height=chart_height,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=170, l=200),
        annotations=[
            dict(
                text="Precision: Accuracy of generated text<br>Recall: Completeness of information<br>F1: Overall quality (balance of precision & recall)",
                xref="paper", yref="paper",
                x=0, y=1.5,
                showarrow=False,
                align="left"
            )
        ]
    )
    
    # Update axes with a wider x-axis range to accommodate markers
    fig.update_xaxes(title_text="Score", range=[-0.1, 1])
    fig.update_yaxes(title_text="Query")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_rouge_scores(insights_df):
    """
    Create visualization for ROUGE scores.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to prepare the data
    rouge_df, has_rouge_data = prepare_rouge_score_data(insights_df)
    
    if not has_rouge_data:
        st.warning("No ROUGE scores available in this dataset.")
        return
    
    # Create figure for ROUGE scores
    fig = go.Figure()
    
    # Add ROUGE score traces
    fig.add_trace(
        go.Bar(
            y=rouge_df['Query'].tolist(),
            x=rouge_df['ROUGE-1'].tolist(),  # Updated to match the new column name
            name='ROUGE-1 F1',
            marker_color='#d62728',
            orientation='h'
        )
    )
    
    fig.add_trace(
        go.Bar(
            y=rouge_df['Query'].tolist(),
            x=rouge_df['ROUGE-2'].tolist(),  # Updated to match the new column name
            name='ROUGE-2 F1',
            marker_color='#9467bd',
            orientation='h'
        )
    )
    
    fig.add_trace(
        go.Bar(
            y=rouge_df['Query'].tolist(),
            x=rouge_df['ROUGE-L'].tolist(),  # Updated to match the new column name
            name='ROUGE-L F1',
            marker_color='#8c564b',
            orientation='h'
        )
    )
        
    # Create a separate trace for correctness indicators
    correct_markers = []
    incorrect_markers = []
    
    for i, correct in enumerate(rouge_df['Is Correct']):
        if correct == 'Correct':
            correct_markers.append(i)
        else:
            incorrect_markers.append(i)
                
    # Add markers for correct answers
    if correct_markers:
        fig.add_trace(
            go.Scatter(
                x=[-0.05] * len(correct_markers),
                y=[rouge_df['Query'].tolist()[i] for i in correct_markers],
                mode='markers+text',
                marker=dict(symbol='circle', color='#4CAF50', size=10),
                text=['✓'] * len(correct_markers),
                textposition='middle center',
                textfont=dict(color='white'),
                name='Correct',
                hoverinfo='none'
            )
        )
        
    # Add markers for incorrect answers
    if incorrect_markers:
        fig.add_trace(
            go.Scatter(
                x=[-0.05] * len(incorrect_markers),
                y=[rouge_df['Query'].tolist()[i] for i in incorrect_markers],
                mode='markers+text',
                marker=dict(symbol='circle', color='#F44336', size=10),
                text=['✗'] * len(incorrect_markers),
                textposition='middle center',
                textfont=dict(color='white'),
                name='Incorrect',
                hoverinfo='none'
            )
        )
        
    # Calculate dynamic height based on number of entries and query text length
    chart_height = calculate_dynamic_chart_height(rouge_df)
    
    # Update layout
    fig.update_layout(
        title="ROUGE Scores <br>(Higher scores indicate better text matching)",
        height=chart_height,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=170, l=200),
        annotations=[
            dict(
                text="ROUGE-1: Word overlap<br>ROUGE-2: Two-word phrase overlap<br>ROUGE-L: Longest common sequence",
                xref="paper", yref="paper",
                x=0, y=1.5,
                showarrow=False,
                align="left"
            )
        ]
    )
    
    # Update axes with a wider x-axis range to accommodate markers
    fig.update_xaxes(title_text="Score", range=[-0.1, 1])
    fig.update_yaxes(title_text="Query")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_gold_context_presence(insights_df):
    """
    Create visualization showing if gold_context is present in the retrieved context list.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Use the data transformation function to prepare the data
    presence_df = prepare_gold_context_presence_data(insights_df)
    
    if presence_df.empty:
        st.warning("Gold context presence data could not be calculated. Please ensure gold_context and context fields are available in the insights data.")
        return
        
    # Calculate presence statistics
    total_queries = len(presence_df)
    
    # Convert position to a format suitable for visualization
    presence_df['Gold Context Present'] = presence_df['Position'].notnull()
    presence_df['Position_Display'] = presence_df['Position'].apply(lambda x: str(int(x)) if pd.notnull(x) else "Not found")
    
    # Compute statistics
    found_count = sum(presence_df['Gold Context Present'])
    found_percentage = found_count / total_queries * 100 if total_queries > 0 else 0
    not_found_count = total_queries - found_count
    
    # Create two separate visualizations in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Create pie chart of gold context presence
        fig_pie = go.Figure()
        fig_pie.add_trace(
            go.Pie(
                labels=['Present', 'Not Present'],
                values=[found_count, not_found_count],
                marker_colors=['#4CAF50', '#F44336'],
                name="Gold Context Presence"
            )
        )
        
        fig_pie.update_layout(
            title=f"Gold Context Presence<br>{found_count}/{total_queries} ({found_percentage:.1f}%)",
            height=350
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Create a horizontal bar chart with position information - only for found contexts
        found_positions = presence_df[presence_df['Gold Context Present']]
        if not found_positions.empty:
            position_counts = found_positions.groupby('Position_Display').size().reset_index(name='Count')
            # Sort by numeric position (convert back to int for sorting, excluding "Not found")
            position_counts = position_counts[position_counts['Position_Display'] != "Not found"]
            if not position_counts.empty:
                position_counts['Position_Numeric'] = position_counts['Position_Display'].astype(int)
                position_counts = position_counts.sort_values('Position_Numeric')
            
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        y=position_counts['Position_Display'],
                        x=position_counts['Count'],
                        orientation='h',
                        marker_color='#2196F3',
                        name="Position in Results",
                        text=position_counts['Count'],
                        textposition='auto'
                    )
                )
                
                fig_bar.update_layout(
                    title="Position in Retrieved Results<br>(when found)",
                    xaxis_title="Count",
                    yaxis_title="Position",
                    height=350
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No gold context found in retrieved results")
        else:
            st.info("No gold context found in retrieved results")
    
    # Add a breakdown by correctness
    st.subheader("Relationship with Answer Correctness")
    
    # Calculate presence rates for correct vs incorrect answers using ALL data
    correct_df = presence_df[presence_df['Is Correct'] == 'Correct']
    incorrect_df = presence_df[presence_df['Is Correct'] == 'Incorrect']
    
    # Proceed with the analysis even if one group is empty
    correct_count = len(correct_df)
    incorrect_count = len(incorrect_df)
    
    if correct_count > 0 or incorrect_count > 0:
        # Calculate presence rates
        correct_presence = correct_df['Gold Context Present'].mean() * 100 if correct_count > 0 else 0
        incorrect_presence = incorrect_df['Gold Context Present'].mean() * 100 if incorrect_count > 0 else 0
        
        # Create data for the grouped bar chart
        categories = []
        values = []
        colors = []
        texts = []
        
        if correct_count > 0:
            categories.append('Correct Answers')
            values.append(correct_presence)
            colors.append('#4CAF50')
            found_correct = correct_df['Gold Context Present'].sum()
            texts.append(f"{correct_presence:.1f}%<br>({found_correct}/{correct_count})")
        
        if incorrect_count > 0:
            categories.append('Incorrect Answers')
            values.append(incorrect_presence)
            colors.append('#F44336')
            found_incorrect = incorrect_df['Gold Context Present'].sum()
            texts.append(f"{incorrect_presence:.1f}%<br>({found_incorrect}/{incorrect_count})")
        
        # Create a grouped bar chart
        fig_corr = go.Figure()
        fig_corr.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=texts,
                textposition='auto',
                hovertemplate='%{x}<br>Gold Context Present: %{text}<extra></extra>'
            )
        )
        
        fig_corr.update_layout(
            title="Gold Context Presence by Answer Correctness",
            yaxis_title="Gold Context Present (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No data available to compare gold context presence between correct and incorrect answers.")
