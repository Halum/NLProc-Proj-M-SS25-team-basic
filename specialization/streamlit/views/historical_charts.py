"""
Historical Charts View

This module provides visualizations for historical performance metrics across multiple evaluation runs.
It shows trends in retrieval accuracy, BERT scores, ROUGE scores, and similarity scores over time.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import data transformation utilities
from specialization.streamlit.utils.historical_analysis import (
    prepare_historical_bert_scores_by_groups,
    prepare_historical_rouge_scores_by_groups,
    prepare_question_correctness_across_iterations
)

def plot_historical_bert_scores(historical_data):
    """
    Create line chart showing BERT scores trend over time, with optional grouping by difficulty or tags.
    """
    # Add a toggle for grouping type
    group_option = st.radio(
        "View Mode:",
        ["All", "Difficulty", "Tags"],
        horizontal=True,
        key="bert_trend_group_by"
    )
    
    if group_option == "All":
        group_by = None
    else:
        group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    scores_by_group, iterations = prepare_historical_bert_scores_by_groups(historical_data, group_by)
    
    if not scores_by_group:
        st.warning("No BERT scores available for visualization.")
        return
    
    fig = go.Figure()
    # Create iteration labels with sample info
    iteration_labels = []
    hover_texts = []
    
    # Generate labels and hover texts with date and sample info
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
    
    # Color scheme for different groups
    colors = px.colors.qualitative.Set3
    
    if group_by is None:
        # Show overall precision, recall, and F1 scores
        scores = scores_by_group['all']
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['precision'],
            mode='lines+markers',
            name='Precision',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>Precision: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['recall'],
            mode='lines+markers',
            name='Recall',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>Recall: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['f1'],
            mode='lines+markers',
            name='F1',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>F1: %{y:.4f}<extra></extra>'
        ))
        
    else:
        # Show F1 scores for each group
        for i, (group, scores) in enumerate(scores_by_group.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=iteration_labels,
                y=scores['f1'],
                mode='lines+markers',
                name=str(group),
                line=dict(color=color, width=2),
                marker=dict(size=8),
                text=hover_texts,
                hovertemplate='%{text}<br>' + str(group) + ' F1: %{y:.4f}<extra></extra>'
            ))
    
    # Calculate y-axis range for auto-zooming
    all_values = []
    for scores in scores_by_group.values():
        all_values.extend(scores['f1'])
        if group_by is None:
            all_values.extend(scores['precision'])
            all_values.extend(scores['recall'])
    
    if all_values:
        y_min = max(0, min(all_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(all_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    title = "BERT Score Trends" if group_by is None else f"BERT F1 Score Trends by {group_option}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Evaluation Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Update x-axis labels
    fig.update_xaxes(
        tickangle=45 if len(iterations) > 5 else 0,
        tickmode='array',
        tickvals=list(range(len(iteration_labels))),
        ticktext=iteration_labels
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
    Create line chart showing ROUGE scores trend over time, with optional grouping by difficulty or tags.
    """
    # Add a toggle for grouping type
    group_option = st.radio(
        "View Mode:",
        ["All", "Difficulty", "Tags"],
        horizontal=True,
        key="rouge_trend_group_by"
    )
    
    if group_option == "All":
        group_by = None
    else:
        group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    scores_by_group, iterations = prepare_historical_rouge_scores_by_groups(historical_data, group_by)
    
    if not scores_by_group:
        st.warning("No ROUGE scores available for visualization.")
        return
    
    fig = go.Figure()
    # Create iteration labels with sample info
    iteration_labels = []
    hover_texts = []
    
    # Generate labels and hover texts with date and sample info
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
    
    # Color scheme for different metrics/groups
    metric_colors = {
        'rouge1': '#1f77b4',
        'rouge2': '#ff7f0e',
        'rougeL': '#2ca02c'
    }
    colors = px.colors.qualitative.Set3
    
    if group_by is None:
        # Show overall ROUGE-1, ROUGE-2, and ROUGE-L scores
        scores = scores_by_group['all']
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['rouge1'],
            mode='lines+markers',
            name='ROUGE-1',
            line=dict(color=metric_colors['rouge1'], width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>ROUGE-1: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['rouge2'],
            mode='lines+markers',
            name='ROUGE-2',
            line=dict(color=metric_colors['rouge2'], width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>ROUGE-2: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=scores['rougeL'],
            mode='lines+markers',
            name='ROUGE-L',
            line=dict(color=metric_colors['rougeL'], width=2),
            marker=dict(size=8),
            text=hover_texts,
            hovertemplate='%{text}<br>ROUGE-L: %{y:.4f}<extra></extra>'
        ))
        
    else:
        # Show ROUGE-1 scores for each group
        for i, (group, scores) in enumerate(scores_by_group.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=iteration_labels,
                y=scores['rouge1'],
                mode='lines+markers',
                name=str(group),
                line=dict(color=color, width=2),
                marker=dict(size=8),
                text=hover_texts,
                hovertemplate='%{text}<br>' + str(group) + ' ROUGE-1: %{y:.4f}<extra></extra>'
            ))
    
    # Calculate y-axis range for auto-zooming
    all_values = []
    for scores in scores_by_group.values():
        all_values.extend(scores['rouge1'])
        if group_by is None:
            all_values.extend(scores['rouge2'])
            all_values.extend(scores['rougeL'])
    
    if all_values:
        y_min = max(0, min(all_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(all_values) + 0.05)  # Add 5% padding above, cap at 1.0
    else:
        y_min, y_max = 0, 1
    
    # Update layout
    title = "ROUGE Score Trends" if group_by is None else f"ROUGE-1 Score Trends by {group_option}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Evaluation Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Add explanation of ROUGE scores
    if group_by is None:
        fig.add_annotation(
            text="ROUGE-1: Word overlap<br>ROUGE-2: Two-word phrase overlap<br>ROUGE-L: Longest common sequence",
            xref="paper", yref="paper",
            x=0, y=1.15,
            showarrow=False,
            align="left"
        )
    
    # Update x-axis labels
    fig.update_xaxes(
        tickangle=45 if len(iterations) > 5 else 0,
        tickmode='array',
        tickvals=list(range(len(iteration_labels))),
        ticktext=iteration_labels
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
        y_min = max(0, min(y_values) - 0.05)  # Add 5% padding below
        y_max = min(1, max(y_values) + 0.05)  # Add 5% padding above, cap at 1.0
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

def plot_question_correctness_across_iterations(historical_data):
    """
    Create a stacked bar chart showing correctness of each question across iterations.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics
    """
    
    # Get the prepared data using the utility function
    chart_data = prepare_question_correctness_across_iterations(historical_data)
    
    # Extract the data components
    question_ids = chart_data['question_ids']
    iteration_data = chart_data['iteration_data']
    iterations = chart_data['iterations']
    
    # If no data is available, simply return
    if not question_ids or not iteration_data:
        st.warning("No question correctness data available for visualization.")
        return
    
    # Show stats about the data
    st.write(f"Found {len(question_ids)} unique questions across {len(iterations)} iterations")
    
    # Create figure
    fig = go.Figure()
    
    # Color scheme for correct/incorrect
    color_correct = '#2ca02c'    # Green
    color_incorrect = '#d62728'  # Red
    
    # For each iteration, add a trace for correct and incorrect answers
    for i, iteration_questions in enumerate(iteration_data):
        # Prepare data for this iteration
        correct_counts = []
        incorrect_counts = []
        
        # For each question, check if it was answered correctly in this iteration
        for q_id in question_ids:
            if q_id in iteration_questions:
                if iteration_questions[q_id]:
                    correct_counts.append(1)  # Correct
                    incorrect_counts.append(0)
                else:
                    correct_counts.append(0)
                    incorrect_counts.append(1)  # Incorrect
            else:
                # Question not present in this iteration
                correct_counts.append(0)
                incorrect_counts.append(0)
        
        # Add trace for correct answers (bottom of stack)
        fig.add_trace(go.Bar(
            x=question_ids,
            y=correct_counts,
            name=f"{iterations[i]} - Correct",
            marker_color=color_correct,  # Consistent green color for all correct answers
            customdata=[[iterations[i]] * len(question_ids)],
            hovertemplate='Question: %{x}<br>Iteration: %{customdata[0]}<br>Status: Correct<extra></extra>'
        ))
        
        # Add trace for incorrect answers (top of stack)
        fig.add_trace(go.Bar(
            x=question_ids,
            y=incorrect_counts,
            name=f"{iterations[i]} - Incorrect",
            marker_color=color_incorrect,  # Consistent red color for all incorrect answers
            customdata=[[iterations[i]] * len(question_ids)],
            hovertemplate='Question: %{x}<br>Iteration: %{customdata[0]}<br>Status: Incorrect<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Question Correctness Across Iterations",
        xaxis_title="Question ID",
        yaxis_title="Result by Iteration",
        barmode='stack',
        hovermode="closest",
        margin=dict(l=50, r=50, t=80, b=100),  # Add more bottom margin for question ID labels
        showlegend=False,  # Remove legends from the chart
        # Set y-axis to only show integer values since iterations are discrete
        yaxis=dict(
            dtick=1,  # Set tick interval to 1
            tick0=0,  # Start ticks at 0
            tickmode='linear',  # Use linear tick mode for even spacing
            tickformat='d'  # Display as integers
        )
    )
    
    # Adjust bar width based on the number of questions
    if len(question_ids) > 5:
        # Calculate width based on number of questions, with thicker bars
        # Increase minimum width from 0.3 to 0.5 and adjust the scaling factor
        bar_width = max(0.5, 2.0 / (len(question_ids) / 20))
        fig.update_traces(width=bar_width)
    
    # Handle x-axis display based on number of questions
    fig.update_xaxes(
        tickangle=90,
        tickmode='array',
        tickvals=question_ids,  # Use actual question IDs as tick values
        ticktext=question_ids,  # And as tick labels
        # Force the category order to follow our sorted question IDs
        categoryorder='array',
        categoryarray=question_ids
    )
    
    # If there are many questions, limit the number of ticks to avoid crowding
    if len(question_ids) > 30:
        fig.update_xaxes(
            nticks=30
        )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_accuracy(historical_data):
    """
    Create a line chart showing accuracy trends and context found percentage over time.
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing historical metrics with timestamps
    """
    # Create iterations for x-axis
    iterations = list(range(1, len(historical_data) + 1))
    iteration_labels = []
    hover_texts_accuracy = []
    hover_texts_context = []
    
    # Check if we have context metrics available
    has_context_metrics = 'context_found_percent' in historical_data.columns and not historical_data['context_found_percent'].isnull().all()
    
    # Create both axis labels and hover text
    for i, (_, row) in enumerate(historical_data.iterrows(), 1):
        correct = row.get('correct_count', 0)
        total = row.get('total_samples', 0)
        date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Format axis label as "Iteration n (correct/total)"
        iteration_label = f"Iteration {i} ({correct}/{total})"
        iteration_labels.append(iteration_label)
        
        # Format hover text for accuracy
        if total > 0:
            percent_correct = (correct / total) * 100
            hover_text_acc = f"Date: {date_str}<br>Accuracy: {percent_correct:.1f}%<br>Correct: {correct}/{total}"
        else:
            hover_text_acc = f"Date: {date_str}<br>Samples: {correct}/{total}"
        hover_texts_accuracy.append(hover_text_acc)
        
        # Format hover text for context found
        if has_context_metrics:
            context_found = row.get('context_found_percent', 0)
            hover_text_ctx = f"Date: {date_str}<br>Context Found: {context_found:.1f}%<br>Samples: {correct}/{total}"
            hover_texts_context.append(hover_text_ctx)
    
    # Create the figure
    fig = go.Figure()
    
    # Add trace for accuracy
    fig.add_trace(go.Scatter(
        x=iteration_labels,
        y=historical_data['accuracy_percent'],
        mode='lines+markers',
        name='Answer Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        text=hover_texts_accuracy,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add trace for context found percentage if available
    if has_context_metrics:
        fig.add_trace(go.Scatter(
            x=iteration_labels,
            y=historical_data['context_found_percent'],
            mode='lines+markers',
            name='Context Found %',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10),
            text=hover_texts_context,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Calculate y-axis range for auto-zooming with padding
    y_values = historical_data['accuracy_percent'].dropna().tolist()
    
    if has_context_metrics:
        y_values.extend(historical_data['context_found_percent'].dropna().tolist())
    
    if y_values:
        y_min = max(0, min(y_values) - 5)  # Subtract 5 percentage points, min 0
        y_max = min(100, max(y_values) + 5)  # Add 5 percentage points, max 100
    else:
        y_min, y_max = 0, 100
    
    # Update layout
    fig.update_layout(
        title="Answer Accuracy & Context Coverage Trend",
        xaxis_title="Evaluation Iteration",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        margin=dict(b=100 if len(iterations) > 5 else 80)  # Add more bottom margin for angled labels
    )
    
    # Update x-axis labels
    fig.update_xaxes(
        tickangle=45 if len(iterations) > 5 else 0,
        tickmode='array',
        tickvals=list(range(len(iteration_labels))),
        ticktext=iteration_labels
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_context_metrics(historical_data):
    """
    Create a chart showing context retrieval position metrics over time:
    Average position of gold context in retrieval results
    
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
            hover_text = f"Date: {date_str}<br>Avg Position: {avg_position:.2f}<br>Found in: {context_found:.1f}%<br>Samples: {correct}/{total}"
        else:
            hover_text = f"Date: {date_str}<br>Avg Position: N/A<br>Found in: {context_found:.1f}%<br>Samples: {correct}/{total}"
        
        hover_texts.append(hover_text)
    
    # Create the figure
    fig = go.Figure()
    
    # Add trace for average context distance (lower is better)
    fig.add_trace(go.Scatter(
        x=iteration_labels,
        y=historical_data['avg_context_distance'],
        mode='lines+markers',
        name='Avg Position',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Calculate y-axis range for auto-zooming
    position_values = historical_data['avg_context_distance'].dropna().tolist()
    
    if position_values:
        y_min = max(0, min(position_values) - 0.5)  # Subtract 0.5, min 0
        y_max = max(position_values) + 0.5  # Add 0.5
    else:
        y_min, y_max = 0, 5
    
    # Update layout
    fig.update_layout(
        title="Context Retrieval Position Trend",
        xaxis_title="Evaluation Iteration",
        yaxis=dict(
            title="Avg Gold Context Position (lower is better)",
            range=[y_min, y_max]
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(iteration_labels))),
            ticktext=iteration_labels,
            tickangle=45 if len(iterations) > 5 else 0
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
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
    
    # Question Correctness Section
    st.subheader("Question Correctness Across Iterations")
    st.write("This chart shows the correctness of each question across different evaluation iterations. Each bar represents a question, stacked by iteration results.")
    
    # Check if we have insight data with id and is_correct fields
    has_question_data = any('insight_data' in row and 
                           isinstance(row['insight_data'], pd.DataFrame) and 
                           not row['insight_data'].empty and
                           'id' in row['insight_data'].columns and 
                           'is_correct' in row['insight_data'].columns
                           for _, row in historical_data.iterrows())
    
    if has_question_data:
        plot_question_correctness_across_iterations(historical_data)
    else:
        st.warning("No detailed question data available for visualization.")
    
    st.markdown("---")
    
    # Accuracy Section
    st.subheader("Answer Accuracy & Context Coverage Trend")
    st.write("This chart shows how the overall accuracy of the RAG system and the percentage of queries where gold context was found have changed over time.")
    
    # Check if we have accuracy metrics
    has_accuracy = 'accuracy_percent' in historical_data.columns and not historical_data['accuracy_percent'].isnull().all()
    
    if has_accuracy:
        plot_historical_accuracy(historical_data)
    else:
        st.warning("No accuracy metrics available in historical data.")
    
    st.markdown("---")
    
    # Context Retrieval Section
    st.subheader("Context Position Trend")
    st.write("This chart shows how the average position of gold context in retrieval results has changed over time (lower is better).")
    
    # Check if we have context metrics
    has_context_metrics = all(col in historical_data.columns for col in ['avg_context_distance', 'context_found_percent'])
    
    if has_context_metrics:
        plot_historical_context_metrics(historical_data)
    else:
        st.warning("No context retrieval metrics available in historical data.")
    
    # End of charts
    st.markdown("---")
