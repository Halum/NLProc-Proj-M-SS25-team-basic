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
    
    # Create iteration labels with sample info
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    
    # Create labels for iterations
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
    
    # Prepare data for box plots
    precision_data = []
    recall_data = []
    f1_data = []
    
    for _, row in historical_data.iterrows():
        insightdf = row['insight_data']
        
        # Extract BERT scores from each insight
        iter_precision = []
        iter_recall = []
        iter_f1 = []
        
        for _, insight in insightdf.iterrows():
            if 'bert_score' in insight and isinstance(insight['bert_score'], dict):
                if 'bert_precision' in insight['bert_score']:
                    iter_precision.append(insight['bert_score']['bert_precision'])
                if 'bert_recall' in insight['bert_score']:
                    iter_recall.append(insight['bert_score']['bert_recall'])
                if 'bert_f1' in insight['bert_score']:
                    iter_f1.append(insight['bert_score']['bert_f1'])
        
        precision_data.append(iter_precision)
        recall_data.append(iter_recall)
        f1_data.append(iter_f1)
    
    # Create subplots for better organization
    fig = go.Figure()
    
    # Add box plots for precision
    for i, data in enumerate(precision_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} Precision',
                marker_color='#1f77b4',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=0,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='Precision: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for recall
    for i, data in enumerate(recall_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} Recall',
                marker_color='#ff7f0e',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=1,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='Recall: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for F1
    for i, data in enumerate(f1_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} F1',
                marker_color='#2ca02c',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=2,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='F1: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add average points as markers for comparison
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
    
    # Create iteration labels with sample info
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    
    # Create labels for iterations
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
    
    # Prepare data for box plots
    rouge1_data = []
    rouge2_data = []
    rougeL_data = []
    
    for _, row in historical_data.iterrows():
        insightdf = row['insight_data']
        
        # Extract ROUGE scores from each insight
        iter_rouge1 = []
        iter_rouge2 = []
        iter_rougeL = []
        
        for _, insight in insightdf.iterrows():
            if 'rouge_score' in insight and isinstance(insight['rouge_score'], dict):
                if 'rouge1_fmeasure' in insight['rouge_score']:
                    iter_rouge1.append(insight['rouge_score']['rouge1_fmeasure'])
                if 'rouge2_fmeasure' in insight['rouge_score']:
                    iter_rouge2.append(insight['rouge_score']['rouge2_fmeasure'])
                if 'rougeL_fmeasure' in insight['rouge_score']:
                    iter_rougeL.append(insight['rouge_score']['rougeL_fmeasure'])
        
        rouge1_data.append(iter_rouge1)
        rouge2_data.append(iter_rouge2)
        rougeL_data.append(iter_rougeL)
    
    # Create figure
    fig = go.Figure()
    
    # Add box plots for ROUGE-1
    for i, data in enumerate(rouge1_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} ROUGE-1',
                marker_color='#d62728',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=0,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='ROUGE-1: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for ROUGE-2
    for i, data in enumerate(rouge2_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} ROUGE-2',
                marker_color='#9467bd',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=1,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='ROUGE-2: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add box plots for ROUGE-L
    for i, data in enumerate(rougeL_data):
        if data:  # Only add if we have data
            fig.add_trace(go.Box(
                y=data,
                x=[iteration_labels[i]] * len(data),
                name=f'Iter {i+1} ROUGE-L',
                marker_color='#8c564b',
                boxmean=True,  # Show mean as a dashed line
                showlegend=False,
                offsetgroup=2,
                customdata=[iteration_labels[i]] * len(data),
                hovertemplate='ROUGE-L: %{y:.4f}<br>%{customdata}<extra></extra>'
            ))
    
    # Add average points as markers for comparison
    # Add a trace for the mean ROUGE-1 values
    fig.add_trace(go.Scatter(
        x=iteration_labels,
        y=historical_data['avg_rouge1_f1'],
        mode='markers+lines',
        name='Mean ROUGE-1',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        legendgroup='mean',
    ))
    
    # Add a trace for the mean ROUGE-2 values
    fig.add_trace(go.Scatter(
        x=iteration_labels,
        y=historical_data['avg_rouge2_f1'],
        mode='markers+lines',
        name='Mean ROUGE-2',
        line=dict(color='#9467bd', width=2, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        legendgroup='mean',
    ))
    
    # Add a trace for the mean ROUGE-L values  
    fig.add_trace(go.Scatter(
        x=iteration_labels,
        y=historical_data['avg_rougeL_f1'],
        mode='markers+lines',
        name='Mean ROUGE-L',
        line=dict(color='#8c564b', width=2, dash='dash'),
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
        title="Historical ROUGE F1 Scores Trend (Average Values)",
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
    
    # Check if we have context distance data
    if 'avg_context_distance' not in data_for_chart.columns or data_for_chart['avg_context_distance'].isnull().all():
        st.warning("No context distance data is available in the historical metrics. This means the gold context wasn't found in the retrieved contexts list.")
        return
    
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
        title="Historical Retrieval Accuracy Trend (Gold Context Position)",
        xaxis_title="Evaluation Iteration (correct/total)",
        yaxis_title="Average Gold Context Position (lower is better)",
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
            
            # Remove insight_data column which contains full dataframes
            if 'insight_data' in display_df.columns:
                display_df = display_df.drop(columns=['insight_data'])
            
            st.dataframe(display_df, use_container_width=True)
    
    # Display each chart section sequentially
    st.markdown("---")
    st.subheader("Retrieval Accuracy by Iteration")
    st.write("This chart shows the average position where the gold context was found in the retrieved contexts list. Lower values indicate better retrieval accuracy, with 1.0 being ideal (gold context found as the first result).")
    
    # Check if we have at least some context distance data
    has_distance_data = ('avg_context_distance' in historical_data.columns and 
                         not historical_data['avg_context_distance'].isnull().all())
    
    if not has_distance_data:
        st.warning("No context distance data is available. This metric requires finding 'gold_context' within the list of retrieved contexts.")
        st.info("To resolve this: ensure your evaluation pipeline includes both 'gold_context' and 'context' (list of retrieved contexts) in the evaluation insights.")
    else:
        plot_historical_retrieval_accuracy(historical_data)
    
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
