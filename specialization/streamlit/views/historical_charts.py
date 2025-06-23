"""
Historical Charts View

This module provides visualizations for historical performance metrics across multiple evaluation runs.
It shows trends in BERT scores, ROUGE scores, and similarity scores over time.
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
