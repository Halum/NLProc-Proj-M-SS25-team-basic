"""
Historical Charts View

This module provides visualizations for historical performance metrics across multiple evaluation runs.
It shows trends in retrieval accuracy, BERT scores, ROUGE scores, and similarity scores over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import data transformation utilities
from specialization.streamlit.utils.data_transformation import extract_score_distributions

def plot_historical_bert_scores(historical_data):
    """
    Create box and whisker plots showing the distribution of BERT scores by iteration.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for BERT scores.")
        return
    
    # Check if we have insight_data available for calculating distributions
    has_insight_data = all('insight_data' in row and isinstance(row['insight_data'], pd.DataFrame) 
                          for _, row in historical_data.iterrows())
    
    if not has_insight_data:
        # Fall back to line chart if we don't have the raw insight data
        st.info("Detailed score distributions not available. Showing average scores only.")
        plot_historical_bert_scores_line(historical_data)
        return
    
    # Extract BERT score distributions and iteration labels using the utility function
    data_arrays, iteration_labels = extract_score_distributions(historical_data, 'bert')
    
    # Unpack the returned data
    if len(data_arrays) == 3:
        precision_data, recall_data, f1_data = data_arrays
    else:
        # Fallback in case data extraction failed
        precision_data, recall_data, f1_data = [], [], []
        
    # Create iterations list for chart building
    iterations = list(range(1, len(historical_data) + 1))
    
    # Create subplots for better organization
    fig = go.Figure()
    
    # Add box plots for precision
    for i, data in enumerate(precision_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} Precision',
                marker_color='#1f77b4',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=0,
                customdata=[label] * len(data),
                hovertemplate='Precision: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for recall
    for i, data in enumerate(recall_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} Recall',
                marker_color='#ff7f0e',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=1,
                customdata=[label] * len(data),
                hovertemplate='Recall: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for F1
    for i, data in enumerate(f1_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} F1',
                marker_color='#2ca02c',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=2,
                customdata=[label] * len(data),
                hovertemplate='F1: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add average points as markers for comparison
    # Check if we have avg_bert metrics
    has_bert_metrics = all(col in historical_data.columns for col in ['avg_bert_precision', 'avg_bert_recall', 'avg_bert_f1'])
    
    if has_bert_metrics and len(historical_data) == len(iteration_labels):
        # Add a trace for the mean precision values
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_bert_precision'],
            mode='markers+lines',
            name='Mean Precision',
            line=dict(color='#1f77b4', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
        
        # Add a trace for the mean recall values
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_bert_recall'],
            mode='markers+lines',
            name='Mean Recall',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
        
        # Add a trace for the mean F1 values  
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_bert_f1'],
            mode='markers+lines',
            name='Mean F1',
            line=dict(color='#2ca02c', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
    elif len(historical_data) != len(iteration_labels):
        st.warning(f"Mismatch between historical data length ({len(historical_data)}) and iteration labels ({len(iteration_labels)})")
    elif not has_bert_metrics:
        st.info("Average BERT metrics not available in historical data")
    
    # Calculate y-axis range for auto-zooming
    all_values = []
    for data_list in precision_data + recall_data + f1_data:
        all_values.extend(data_list)
    
    if all_values:
        y_min = max(0, min(all_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(all_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="BERT Score Distributions by Iteration",
        xaxis_title="Evaluation Iteration",
        yaxis_title="Score",
        boxmode='group',  # Group boxes for each iteration
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(b=100 if iterations and len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Add annotations to indicate what each box represents
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text="Each box shows score distribution<br>Diamond markers show mean values",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_bert_scores_line(historical_data):
    """
    Fallback function to create a line chart showing average BERT scores by iteration.
    Used when detailed insight data is not available.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
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
        title="Historical BERT Scores Trend (Average Values)",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Score",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=0 if iterations and len(iterations) <= 5 else 45
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(b=100 if iterations and len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_rouge_scores(historical_data):
    """
    Create box and whisker plots showing the distribution of ROUGE scores by iteration.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    if historical_data.empty:
        st.warning("No historical data available for ROUGE scores.")
        return
    
    # Check if we have insight_data available for calculating distributions
    has_insight_data = all('insight_data' in row and isinstance(row['insight_data'], pd.DataFrame) 
                          for _, row in historical_data.iterrows())
    
    if not has_insight_data:
        # Fall back to line chart if we don't have the raw insight data
        st.info("Detailed ROUGE score distributions not available. Showing average scores only.")
        plot_historical_rouge_scores_line(historical_data)
        return
    
    # Extract ROUGE score distributions and iteration labels using the utility function
    data_arrays, iteration_labels = extract_score_distributions(historical_data, 'rouge')
    
    # Unpack the returned data
    if len(data_arrays) == 3:
        rouge1_data, rouge2_data, rougeL_data = data_arrays
    else:
        # Fallback in case data extraction failed
        st.info("Could not extract detailed ROUGE score distributions. Showing average scores only.")
        plot_historical_rouge_scores_line(historical_data)
        return
    
    # Check if we have actual data in the arrays (not just empty lists)
    if all(not any(x) for x in rouge1_data + rouge2_data + rougeL_data):
        st.info("No detailed ROUGE scores found in the insight data. Showing average scores only.")
        plot_historical_rouge_scores_line(historical_data)
        return
        
    # Create iterations list for chart building
    iterations = list(range(1, len(historical_data) + 1))
    
    # Create subplots for better organization
    fig = go.Figure()
    
    # Add box plots for ROUGE-1
    for i, data in enumerate(rouge1_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} ROUGE-1',
                marker_color='#1f77b4',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=0,
                customdata=[label] * len(data),
                hovertemplate='ROUGE-1: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for ROUGE-2
    for i, data in enumerate(rouge2_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} ROUGE-2',
                marker_color='#ff7f0e',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=1,
                customdata=[label] * len(data),
                hovertemplate='ROUGE-2: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for ROUGE-L
    for i, data in enumerate(rougeL_data):
        if data and i < len(iteration_labels):  # Check both data and index
            label = iteration_labels[i]
            fig.add_trace(go.Box(
                y=data,
                x=[label] * len(data),
                name=f'Iter {i+1} ROUGE-L',
                marker_color='#2ca02c',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=2,
                customdata=[label] * len(data),
                hovertemplate='ROUGE-L: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add average points as markers for comparison
    # Check if we have avg_rouge metrics
    has_rouge_metrics = all(col in historical_data.columns for col in 
                          ['avg_rouge1_f1', 'avg_rouge2_f1', 'avg_rougeL_f1'])
    
    if has_rouge_metrics and len(historical_data) == len(iteration_labels):
        # Add a trace for the mean ROUGE-1 values
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_rouge1_f1'],
            mode='markers+lines',
            name='Mean ROUGE-1',
            line=dict(color='#1f77b4', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
        
        # Add a trace for the mean ROUGE-2 values
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_rouge2_f1'],
            mode='markers+lines',
            name='Mean ROUGE-2',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
        
        # Add a trace for the mean ROUGE-L values
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['avg_rougeL_f1'],
            mode='markers+lines',
            name='Mean ROUGE-L',
            line=dict(color='#2ca02c', width=2, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            legendgroup='mean',
        ))
    
    # Calculate y-axis range for auto-zooming
    all_values = []
    for data_list in rouge1_data + rouge2_data + rougeL_data:
        all_values.extend(data_list)
    
    if all_values:
        y_min = max(0, min(all_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(all_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="ROUGE Score Distributions by Iteration",
        xaxis_title="Evaluation Iteration",
        yaxis_title="Score",
        boxmode='group',  # Group boxes for each iteration
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Add annotations to indicate what each box represents
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text="Each box shows score distribution<br>Diamond markers show mean values",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_rouge_scores_line(historical_data):
    """
    Fallback function to create a line chart showing average ROUGE scores by iteration.
    Used when detailed insight data is not available.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
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
    
    # Check if we have ROUGE metrics available
    has_rouge1 = 'avg_rouge1_f1' in historical_data.columns and not historical_data['avg_rouge1_f1'].isnull().all()
    has_rouge2 = 'avg_rouge2_f1' in historical_data.columns and not historical_data['avg_rouge2_f1'].isnull().all()
    has_rougeL = 'avg_rougeL_f1' in historical_data.columns and not historical_data['avg_rougeL_f1'].isnull().all()
    
    if has_rouge1:
        # Add trace for ROUGE-1
        fig.add_trace(go.Scatter(
            x=iterations,
            y=historical_data['avg_rouge1_f1'],
            mode='lines+markers',
            name='ROUGE-1',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='Iteration %{x}<br>ROUGE-1: %{y:.4f}<br>%{text}<extra></extra>'
        ))
    
    if has_rouge2:
        # Add trace for ROUGE-2
        fig.add_trace(go.Scatter(
            x=iterations,
            y=historical_data['avg_rouge2_f1'],
            mode='lines+markers',
            name='ROUGE-2',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='Iteration %{x}<br>ROUGE-2: %{y:.4f}<br>%{text}<extra></extra>'
        ))
    
    if has_rougeL:
        # Add trace for ROUGE-L
        fig.add_trace(go.Scatter(
            x=iterations,
            y=historical_data['avg_rougeL_f1'],
            mode='lines+markers',
            name='ROUGE-L',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='Iteration %{x}<br>ROUGE-L: %{y:.4f}<br>%{text}<extra></extra>'
        ))
    
    # Calculate y-axis range for auto-zooming
    y_values = []
    for col in ['avg_rouge1_f1', 'avg_rouge2_f1', 'avg_rougeL_f1']:
        if col in historical_data.columns and not historical_data[col].isnull().all():
            y_values.extend(historical_data[col].dropna().tolist())
    
    if y_values:
        y_min = max(0, min(y_values) - 0.05)  # Add 5% padding below, min 0
        y_max = min(1, max(y_values) + 0.05)  # Add 5% padding above, max 1
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    fig.update_layout(
        title="ROUGE Scores by Iteration",
        xaxis_title="Evaluation Iteration",
        yaxis_title="F1 Score",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=45 if len(iterations) > 5 else 0
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_accuracy(historical_data):
    """
    Create a line chart showing accuracy trends over time.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    # Create iterations for x-axis
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
            hover_text = f"Date: {date_str}<br>Accuracy: {percent_correct:.1f}%<br>Correct: {correct}/{total}"
        else:
            hover_text = f"Date: {date_str}<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    # Create the figure
    fig = go.Figure()
    
    # Add trace for accuracy
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['accuracy_percent'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Accuracy: %{y:.1f}%<br>%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming with padding
    y_values = historical_data['accuracy_percent'].dropna().tolist()
    
    if y_values:
        y_min = max(0, min(y_values) - 5)  # Subtract 5 percentage points, min 0
        y_max = min(100, max(y_values) + 5)  # Add 5 percentage points, max 100
    else:
        y_min, y_max = 0, 100
    
    # Update layout
    fig.update_layout(
        title="Answer Accuracy Trend",
        xaxis_title="Evaluation Iteration",
        yaxis_title="Accuracy (%)",
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=45 if len(iterations) > 5 else 0
        ),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_context_metrics(historical_data):
    """
    Create a dual-axis chart showing context retrieval metrics over time:
    1. Average position of gold context
    2. Percentage of queries where gold context was found
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    # Create iterations for x-axis
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
        context_found = row.get('context_found_percent', 0)
        avg_position = row.get('avg_context_distance', None)
        
        if avg_position is not None:
            hover_text = f"Date: {date_str}<br>Found in: {context_found:.1f}%<br>Avg Position: {avg_position:.2f}"
        else:
            hover_text = f"Date: {date_str}<br>Found in: {context_found:.1f}%<br>Avg Position: N/A"
        
        hover_texts.append(hover_text)
    
    # Create a figure with secondary y-axis
    fig = go.Figure()
    
    # Add trace for average context distance (lower is better)
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['avg_context_distance'],
        mode='lines+markers',
        name='Avg Position',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Avg Position: %{y:.2f}<br>%{text}<extra></extra>'
    ))
    
    # Add trace for context found percentage
    fig.add_trace(go.Scatter(
        x=iterations,
        y=historical_data['context_found_percent'],
        mode='lines+markers',
        name='Found %',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='Iteration %{x}<br>Found: %{y:.1f}%<br>%{text}<extra></extra>',
        yaxis="y2"  # Use secondary y-axis
    ))
    
    # Calculate y-axis range for auto-zooming
    position_values = historical_data['avg_context_distance'].dropna().tolist()
    percent_values = historical_data['context_found_percent'].dropna().tolist()
    
    if position_values:
        y1_min = max(0, min(position_values) - 0.5)  # Subtract 0.5, min 0
        y1_max = max(position_values) + 0.5  # Add 0.5
    else:
        y1_min, y1_max = 0, 5
    
    if percent_values:
        y2_min = max(0, min(percent_values) - 5)  # Subtract 5 percentage points, min 0
        y2_max = min(100, max(percent_values) + 5)  # Add 5 percentage points, max 100
    else:
        y2_min, y2_max = 0, 100
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title="Context Retrieval Performance Trend",
        xaxis_title="Evaluation Iteration",
        yaxis=dict(
            title="Avg Gold Context Position (lower is better)",
            range=[y1_min, y1_max],
            side="left"
        ),
        yaxis2=dict(
            title="Context Found (%)",
            range=[y2_min, y2_max],
            side="right",
            overlaying="y",
            tickmode="auto"
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=iterations,
            ticktext=iteration_labels,
            tickangle=45 if len(iterations) > 5 else 0
        ),
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
        )
    )
    
    # Add annotation for the reference line
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=1,
        text="Ideal position (1)",
        showarrow=False,
        font=dict(
            color="green",
            size=10
        ),
        bgcolor="rgba(255, 255, 255, 0.7)"
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
        st.info(f"📊 {len(historical_data)} evaluation iterations (from {earliest} to {latest})")
    
    # Display the raw data table if there's data
    if not historical_data.empty:
        with st.expander("View Raw Historical Data", expanded=False):
            # Format timestamp for display
            display_df = historical_data.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Round numeric columns to 4 decimal places for better readability
            numeric_cols = display_df.select_dtypes(include=['float64']).columns
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            # Remove insight_data column which contains full dataframes
            if 'insight_data' in display_df.columns:
                display_df = display_df.drop(columns=['insight_data'])
            
            st.dataframe(display_df, use_container_width=True)
    
    # Display each chart section sequentially
    st.markdown("---")
    
    # BERT Scores Section
    st.subheader("BERT Scores Trend")
    st.write("These charts show how BERT scores (precision, recall, and F1) have evolved across evaluation runs.")
    
    # Check if we have BERT metrics
    has_bert_metrics = all(col in historical_data.columns for col in ['avg_bert_precision', 'avg_bert_recall', 'avg_bert_f1'])
    
    if has_bert_metrics:
        plot_historical_bert_scores(historical_data)
    else:
        st.warning("No BERT score metrics available in historical data.")
    
    st.markdown("---")
    
    # ROUGE Scores Section
    st.subheader("ROUGE Scores Trend")
    st.write("These charts show how ROUGE scores (measuring text overlap) have evolved across evaluation runs.")
    
    # Check if we have ROUGE metrics - any of the average metrics
    has_rouge_metrics = any(col in historical_data.columns and not historical_data[col].isnull().all() 
                          for col in ['avg_rouge1_f1', 'avg_rouge2_f1', 'avg_rougeL_f1'])
    
    if has_rouge_metrics:
        # Try to use the box plot visualization first, it will fall back to line chart if needed
        plot_historical_rouge_scores(historical_data)
    else:
        st.warning("No ROUGE score metrics available in historical data.")
    
    st.markdown("---")
    
    # Accuracy Section
    st.subheader("Answer Accuracy Trend")
    st.write("This chart shows how the overall accuracy of the RAG system has changed over time.")
    
    # Check if we have accuracy metrics
    has_accuracy = 'accuracy_percent' in historical_data.columns and not historical_data['accuracy_percent'].isnull().all()
    
    if has_accuracy:
        plot_historical_accuracy(historical_data)
    else:
        st.warning("No accuracy metrics available in historical data.")
    
    st.markdown("---")
    
    # Context Retrieval Section
    st.subheader("Context Retrieval Trend")
    st.write("This chart shows how retrieval performance has changed over time, measured by gold context position and presence.")
    
    # Check if we have context metrics
    has_context_metrics = all(col in historical_data.columns for col in ['avg_context_distance', 'context_found_percent'])
    
    if has_context_metrics:
        plot_historical_context_metrics(historical_data)
    else:
        st.warning("No context retrieval metrics available in historical data.")
    
    st.markdown("---")
    
    # Customer requested to remove detailed context position distribution charts
