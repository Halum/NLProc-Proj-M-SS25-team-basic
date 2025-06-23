"""
Historical Charts View

This module provides visualizations for historical performance metrics across multiple evaluation runs.
It shows trends in retrieval accuracy, BERT scores, ROUGE scores, and similarity scores over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def plot_historical_bert_scores(historical_data):
    """
    Create a line chart showing average BERT scores by iteration.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for BERT scores.")
        return
        
    # Create iteration labels with sample info
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    hover_texts = []
    
    # Create both axis labels and hover text
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
        
        # Format hover text with additional details
        if total > 0:
            percent_correct = (correct / total) * 100
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total} ({percent_correct:.1f}% correct)"
        else:
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    # Add traces for each BERT metric
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_bert_precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Precision: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_bert_recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Recall: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_bert_f1'],
        mode='lines+markers',
        name='F1',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>F1: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming
    y_values = []
    for col in ['avg_bert_precision', 'avg_bert_recall', 'avg_bert_f1']:
        if col in historical_data.columns:
            y_values.extend(historical_data[col].dropna().tolist())
    
    if y_values:
        y_min = max(0, min(y_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(y_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="Historical BERT Scores Trend",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Score",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=0 if len(iterations) <= 5 else 45
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_rouge_scores(historical_data):
    """
    Create a line chart showing average ROUGE F1 scores by iteration.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for ROUGE scores.")
        return
        
    # Create iteration labels with sample info
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    hover_texts = []
    
    # Create both axis labels and hover text
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
        
        # Format hover text with additional details
        if total > 0:
            percent_correct = (correct / total) * 100
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total} ({percent_correct:.1f}% correct)"
        else:
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    # Add traces for each ROUGE F1 metric
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_rouge1_f1'],
        mode='lines+markers',
        name='ROUGE-1 F1',
        line=dict(color='#d62728', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>ROUGE-1 F1: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_rouge2_f1'],
        mode='lines+markers',
        name='ROUGE-2 F1',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>ROUGE-2 F1: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_rougeL_f1'],
        mode='lines+markers',
        name='ROUGE-L F1',
        line=dict(color='#8c564b', width=2),
        marker=dict(size=8),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>ROUGE-L F1: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming
    y_values = []
    for col in ['avg_rouge1_f1', 'avg_rouge2_f1', 'avg_rougeL_f1']:
        if col in historical_data.columns:
            y_values.extend(historical_data[col].dropna().tolist())
    
    if y_values:
        y_min = max(0, min(y_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(y_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="Historical ROUGE F1 Scores Trend",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Score",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=0 if len(iterations) <= 5 else 45
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_similarity_scores(historical_data):
    """
    Create a line chart showing average similarity scores by iteration.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for similarity scores.")
        return
        
    # Create iteration labels with sample info
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    hover_texts = []
    
    # Create both axis labels and hover text
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
        
        # Format hover text with additional details
        if total > 0:
            percent_correct = (correct / total) * 100
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total} ({percent_correct:.1f}% correct)"
        else:
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    # Add trace for similarity scores
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_similarity'],
        mode='lines+markers',
        name='Avg Similarity',
        line=dict(color='#17becf', width=2),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Avg Similarity: %{y:.4f}<br>%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming
    y_values = historical_data['avg_similarity'].dropna().tolist()
    
    if y_values:
        y_min = max(0, min(y_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(y_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="Historical Similarity Scores Trend",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Similarity Score",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=0 if len(iterations) <= 5 else 45
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_retrieval_accuracy(historical_data):
    """
    Create a line chart showing average context distance (retrieval accuracy) by iteration.
    Lower values indicate better retrieval accuracy as context was found closer to the top.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for retrieval accuracy.")
        return
    
    # Make a copy to avoid modifying the original DataFrame
    data_for_chart = historical_data.copy()
    
    # Check if we need to calculate the average context distance
    if 'avg_context_distance' not in data_for_chart.columns:
        # If the column doesn't exist, we'll calculate it
        st.info("Calculating average context distance from retrieved chunk position data...")
        
        # Check if we have individual insight data files stored that we could load
        # This would be implementation-specific, so we'll use the metrics we have
        
        # Check if we have retrieved_chunk_rank or retrieved_positions column
        if 'avg_retrieved_position' in data_for_chart.columns:
            # If we already have the average position, use it directly
            data_for_chart['avg_context_distance'] = data_for_chart['avg_retrieved_position']
        
        # If there's insight_data available for calculations
        elif 'insight_data' in data_for_chart.columns and isinstance(data_for_chart.iloc[0].get('insight_data'), pd.DataFrame):
            # Try to calculate from insight data
            distances = []
            
            for _, row in data_for_chart.iterrows():
                avg_distance = calculate_avg_context_distance(row['insight_data'])
                distances.append(avg_distance if avg_distance is not None else float('nan'))
            
            data_for_chart['avg_context_distance'] = distances
        else:
            # Default value if we can't calculate it
            data_for_chart['avg_context_distance'] = [3.0] * len(data_for_chart)
            st.warning("Could not calculate average context distance. Using placeholder values of 3.0.")
    
    # Create iteration labels with sample info
    iterations = list(range(1, len(data_for_chart) + 1))
    iteration_labels = []
    hover_texts = []
    
    # Create both axis labels and hover text
    for i, (_, row) in enumerate(data_for_chart.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
        
        # Format hover text with additional details
        if total > 0:
            percent_correct = (correct / total) * 100
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total} ({percent_correct:.1f}% correct)"
        else:
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    # Add trace for retrieval accuracy (context distance)
    fig.add_trace(go.Scatter(
        x=iterations,
        y=data_for_chart['avg_context_distance'],
        mode='lines+markers',
        name='Avg Context Distance',
        line=dict(color='#e377c2', width=2),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Avg Context Distance: %{y:.2f}<br>%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming - start from 0, but we'll set a reasonable upper limit
    y_values = data_for_chart['avg_context_distance'].dropna().tolist()
    
    if y_values:
        y_min = 0  # Always start from 0
        y_max = max(y_values) * 1.2  # Add 20% padding above
    else:
        y_min, y_max = 0, 5  # Default range if no data
    
    # Update layout
    fig.update_layout(
        title="Historical Retrieval Accuracy Trend (Average Context Distance)",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Context Distance (lower is better)",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=0 if len(iterations) <= 5 else 45
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Add a horizontal reference line representing "perfect" retrieval (context at position 1)
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=1,
        x1=1,
        y1=1,
        line=dict(
            color="green",
            width=1,
            dash="dash",
        ),
        name="Perfect Retrieval"
    )
    
    # Add annotation for the reference line
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=1,
        text="Ideal (position 1)",
        showarrow=False,
        font=dict(
            color="green",
            size=10
        ),
        bgcolor="rgba(255, 255, 255, 0.7)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_avg_context_distance(insights_df):
    """
    Calculate the average context distance from retrieved_chunk_rank values.
    
    The context distance is the position where the relevant context was found.
    For example, if contexts are found at positions 5, 5, 1, 4, 4, 1, the 
    average context distance would be (5+5+1+4+4+1)/6 = 3.33.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
        
    Returns:
        float: The calculated average context distance, or None if calculation fails
    """
    try:
        # Filter out rows where context wasn't found (position = -1)
        if 'retrieved_chunk_rank' in insights_df.columns:
            context_found = insights_df[insights_df['retrieved_chunk_rank'] >= 0]
            
            if not context_found.empty:
                # Calculate the average distance/position
                avg_distance = context_found['retrieved_chunk_rank'].mean()
                return avg_distance
        
        return None
    except Exception as e:
        st.warning(f"Error calculating average context distance: {str(e)}")
        return None

def display_historical_charts(historical_data):
    """
    Display all historical performance charts.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    st.write("These charts show the trend of key performance metrics across multiple evaluation runs.")
    
    # Add a warning if there are fewer than 2 data points
    if len(historical_data) < 2:
        st.warning("Limited historical data available. Trends may not be meaningful with fewer than 2 evaluation runs.")
    
    # Display the number of iterations
    if not historical_data.empty:
        earliest = historical_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
        latest = historical_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"� {len(historical_data)} evaluation iterations (from {earliest} to {latest})")
    
    # Display the raw data table if there's data
    if not historical_data.empty:
        with st.expander("View Raw Historical Data", expanded=False):
            # Format timestamp for display
            display_df = historical_data.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Round numeric columns to 4 decimal places for better readability
            numeric_cols = display_df.select_dtypes(include=['float64']).columns
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            st.dataframe(display_df, use_container_width=True)
    
    # Display each chart section sequentially
    st.markdown("---")
    st.subheader("Retrieval Accuracy by Iteration")
    st.write("This chart shows the average context distance (position) when found across all evaluation iterations. Lower values indicate better retrieval accuracy as relevant context was found closer to the top of the search results.")
    
    # We'll always try to display the chart, and let the function handle the calculation or warnings
    try:
        plot_historical_retrieval_accuracy(historical_data)
    except Exception as e:
        st.error(f"Error displaying retrieval accuracy chart: {str(e)}")
        st.warning("Could not generate the retrieval accuracy chart. This may indicate missing data or a format issue.")
    
    st.markdown("---")
    st.subheader("BERT Scores by Iteration")
    st.write("This chart shows the average BERT precision, recall, and F1 scores across all evaluation iterations.")
    plot_historical_bert_scores(historical_data)
    
    st.markdown("---")
    st.subheader("ROUGE Scores by Iteration")
    st.write("This chart shows the average ROUGE F1 scores across all evaluation iterations.")
    plot_historical_rouge_scores(historical_data)
    
    st.markdown("---")
    st.subheader("Similarity Scores by Iteration")
    st.write("This chart shows the average similarity scores across all evaluation iterations.")
    plot_historical_similarity_scores(historical_data)
    
    st.markdown("---")
    st.subheader("Retrieval Accuracy by Iteration")
    st.write("This chart shows the average context distance, indicating retrieval accuracy, across all evaluation iterations. Lower values are better.")
    plot_historical_retrieval_accuracy(historical_data)
