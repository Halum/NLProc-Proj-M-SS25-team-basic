"""
Visualization components for RAG performance metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Import data transformation utilities
from specialization.streamlit.utils import (
    prepare_correctness_by_groups_data,
    prepare_bert_score_by_groups_data,
    prepare_rouge_score_by_groups_data
)
from specialization.streamlit.utils.common import prepare_common_grouping

def plot_answer_correctness(insights_df):
    """
    Create a visualization for answer correctness grouped by difficulty or tags.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="correctness_group_by"
    )
    
    group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    # Use the data transformation function to prepare the data
    stacked_data = prepare_correctness_by_groups_data(insights_df, group_by)
    
    if stacked_data.empty:
        st.warning(f"No {group_option.lower()} data available for visualization.")
        return
    
    # Create color map for consistent colors
    color_map = {
        'Correct': '#4CAF50',
        'Incorrect': '#F44336'
    }
    
    # Create the stacked bar chart
    fig = go.Figure()
    
    # Add Correct bars
    fig.add_trace(go.Bar(
        x=stacked_data['Group'],
        y=stacked_data['Correct'],
        name='Correct',
        marker_color=color_map['Correct'],
        hovertemplate='%{y} Correct<extra></extra>'
    ))
    
    # Add Incorrect bars
    fig.add_trace(go.Bar(
        x=stacked_data['Group'],
        y=stacked_data['Incorrect'],
        name='Incorrect',
        marker_color=color_map['Incorrect'],
        hovertemplate='%{y} Incorrect<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Answer Correctness by {group_option}',
        barmode='stack',
        xaxis_title=group_option,
        yaxis_title='Count',
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    

    
def plot_bert_scores(insights_df):
    """
    Create visualization for BERT scores grouped by difficulty or tags.
    Shows Precision, Recall, and F1 scores for each group.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="bert_group_by"
    )
    
    group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    # Get the prepared data
    grouped_df = prepare_bert_score_by_groups_data(insights_df, group_by)
    
    if grouped_df.empty:
        st.warning(f"No BERT scores available for visualization by {group_option.lower()}.")
        return
        
    # Create the vertical bar chart
    fig = go.Figure()
    
    # Define colors for consistency
    color_map = {
        'BERT Precision': '#1f77b4',
        'BERT Recall': '#ff7f0e',
        'BERT F1': '#2ca02c'
    }
    
    # Add bars for each BERT metric
    for metric_type in ['BERT Precision', 'BERT Recall', 'BERT F1']:
        metric_df = grouped_df[grouped_df['Type'] == metric_type]
        if not metric_df.empty:
            fig.add_trace(go.Bar(
                x=metric_df['Group'],
                y=metric_df['Score'],
                name=metric_type,
                marker_color=color_map[metric_type],
                text=[f"{score:.3f}<br>n={count}" for score, count in zip(metric_df['Score'], metric_df['Count'])],
                textposition='auto',
                hovertemplate='%{x}<br>' + metric_type + ': %{y:.3f}<br>Sample size: %{text}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Average BERT Scores by {group_option}",
        xaxis_title=group_option,
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, l=50, r=50, b=100),
        annotations=[
            dict(
                text="BERT scores measure semantic similarity between generated and gold standard answers",
                xref="paper", yref="paper",
                x=0, y=1.15,
                showarrow=False,
                align="left"
            )
        ]
    )
    
    # Rotate x-axis labels if there are many groups
    if len(grouped_df['Group'].unique()) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_rouge_scores(insights_df):
    """
    Create visualization for ROUGE scores grouped by difficulty or tags.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="rouge_group_by"
    )
    
    group_by = 'difficulty' if group_option == "Difficulty" else 'tags'
    
    # Get the prepared data
    grouped_df = prepare_rouge_score_by_groups_data(insights_df, group_by)
    
    if grouped_df.empty:
        st.warning(f"No ROUGE scores available for visualization by {group_option.lower()}.")
        return
        
    # Create the vertical bar chart
    fig = go.Figure()
    
    # Define colors for consistency
    color_map = {
        'ROUGE-1': '#d62728',
        'ROUGE-2': '#9467bd',
        'ROUGE-L': '#8c564b'
    }
    
    # Add bars for each ROUGE metric
    for rouge_type in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        metric_df = grouped_df[grouped_df['Type'] == rouge_type]
        if not metric_df.empty:
            fig.add_trace(go.Bar(
                x=metric_df['Group'],
                y=metric_df['Score'],
                name=rouge_type,
                marker_color=color_map[rouge_type],
                text=[f"{score:.3f}<br>n={count}" for score, count in zip(metric_df['Score'], metric_df['Count'])],
                textposition='auto',
                hovertemplate='%{x}<br>' + rouge_type + ': %{y:.3f}<br>Sample size: %{text}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Average ROUGE Scores by {group_option}",
        xaxis_title=group_option,
        yaxis_title="ROUGE Score",
        yaxis=dict(range=[0, 1]),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, l=50, r=50, b=100),
        annotations=[
            dict(
                text="ROUGE-1: Word overlap<br>ROUGE-2: Two-word phrase overlap<br>ROUGE-L: Longest common sequence",
                xref="paper", yref="paper",
                x=0, y=1.15,
                showarrow=False,
                align="left"
            )
        ]
    )
    
    # Rotate x-axis labels if there are many groups
    if len(grouped_df['Group'].unique()) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_gold_context_presence(insights_df):
    """
    Create visualization showing if gold_context is present in the retrieved context list.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    if 'gold_context_pos' not in insights_df.columns:
        st.warning("Gold context position data not found in the insights.")
        return
        
    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="presence_group_by"
    )
    
    # Use common grouping logic
    insights_df_grouped, actual_group = prepare_common_grouping(insights_df.copy(), group_option)
    
    # Determine context presence based on gold_context_pos field
    insights_df_grouped['context_found'] = insights_df_grouped['gold_context_pos'] > 0
    
    # Group by the selected option and calculate presence
    grouped_data = (
        insights_df_grouped.groupby(actual_group)['context_found']
        .agg(['sum', 'size'])
        .reset_index()
        .rename(columns={'sum': 'Found', 'size': 'Total'})
    )
    
    grouped_data['Not Found'] = grouped_data['Total'] - grouped_data['Found']
    
    if grouped_data.empty:
        st.warning(f"No data available for grouping by {group_option}")
        return

    # Sort by Found count in descending order
    grouped_data = grouped_data.sort_values('Found', ascending=False)

    # Create stacked bar chart
    fig = go.Figure()

    # Add Found bars
    fig.add_trace(go.Bar(
        x=grouped_data[actual_group],
        y=grouped_data['Found'],
        name='Found',
        marker_color='#4CAF50',
        text=grouped_data.apply(lambda x: f"{x['Found']}/{x['Total']}", axis=1),
        textposition='auto',
        hovertemplate='%{x}<br>Found: %{y}<br>%{text}<extra></extra>'
    ))
    
    # Add Not Found bars
    fig.add_trace(go.Bar(
        x=grouped_data[actual_group],
        y=grouped_data['Not Found'],
        name='Not Found',
        marker_color='#F44336',
        text=grouped_data.apply(lambda x: f"{x['Not Found']}/{x['Total']}", axis=1),
        textposition='auto',
        hovertemplate='%{x}<br>Not Found: %{y}<br>%{text}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'Gold Context Presence by {group_option}',
        barmode='stack',
        xaxis_title=group_option,
        yaxis_title='Count',
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_position_distribution(insights_df):
    """
    Create visualization showing the distribution of gold context positions in retrieved results.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    if 'gold_context_pos' not in insights_df.columns:
        st.warning("Gold context position data not found in the insights.")
        return

    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="position_group_by"
    )
    
    # Use common grouping logic
    insights_df_grouped, actual_group = prepare_common_grouping(insights_df.copy(), group_option)
    
    # Filter for found contexts
    position_data = insights_df_grouped[insights_df_grouped['gold_context_pos'] > 0]
    
    if position_data.empty:
        st.warning("No gold context found in retrieved results")
        return
        
    # Group by both the selected option and position
    position_data = (
        position_data.groupby([actual_group, 'gold_context_pos'])
        .size()
        .reset_index(name='count')
    )
    
    if position_data.empty:
        st.warning("No gold context found in retrieved results")
        return

    # Sort groups by total count for consistent coloring
    group_totals = position_data.groupby(actual_group)['count'].sum().sort_values(ascending=False)
    
    # Create color scale for groups
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_map = {group: colors[i % len(colors)] for i, group in enumerate(group_totals.index)}

    # Create figure
    fig = go.Figure()

    # Add a bar trace for each group
    for group in group_totals.index:
        group_data = position_data[position_data[actual_group] == group]
        
        fig.add_trace(go.Bar(
            x=group_data['gold_context_pos'],
            y=group_data['count'],
            name=group,
            marker_color=color_map[group],
            text=group_data['count'],
            textposition='auto',
            hovertemplate=(
                f"{group}<br>" +
                "Position: %{x}<br>" +
                "Count: %{y}<br>" +
                "<extra></extra>"
            )
        ))

    # Update layout
    fig.update_layout(
        title=f"Position Distribution by {group_option}",
        barmode='group',
        xaxis_title="Position in Retrieved Results",
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis_title="Count",
        showlegend=True,
        legend_title=group_option,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        bargap=0.2,
        bargroupgap=0.1
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_presence_by_correctness(insights_df):
    """
    Create visualization showing gold context presence rates by answer correctness.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    if not all(col in insights_df.columns for col in ['gold_context_pos', 'is_correct']):
        st.warning("Required data (gold_context_pos or is_correct) not found in the insights.")
        return

    # Add a toggle to switch between difficulty and tags
    group_option = st.radio(
        "Group by:",
        ["Difficulty", "Tags"],
        horizontal=True,
        key="presence_correctness_group_by"
    )
    
    # Use common grouping logic
    insights_df_grouped, actual_group = prepare_common_grouping(insights_df.copy(), group_option)
    
    # Determine context presence
    insights_df_grouped['context_found'] = insights_df_grouped['gold_context_pos'] > 0
    
    # Group by both the selected option and correctness
    grouped_data = (
        insights_df_grouped.groupby([actual_group, 'is_correct'])['context_found']
        .agg(['sum', 'size'])
        .reset_index()
        .rename(columns={'sum': 'Found', 'size': 'Total'})
    )
    
    if grouped_data.empty:
        st.warning(f"No data available for grouping by {group_option}")
        return

    # Get the correct answers data and sort it by Found count
    correct_data = grouped_data[grouped_data['is_correct']].sort_values('Found', ascending=False)
    # Get the incorrect answers data in the same order as correct data
    incorrect_data = grouped_data[~grouped_data['is_correct']]
    incorrect_data = incorrect_data.set_index(actual_group).reindex(
        correct_data[actual_group]
    ).reset_index()

    # Create figure
    fig = go.Figure()

    # Add bars for correct answers
    fig.add_trace(go.Bar(
        x=correct_data[actual_group],
        y=correct_data['Found'],
        name='Correct Answers',
        marker_color='#4CAF50',
        text=correct_data.apply(lambda x: f"{x['Found']}/{x['Total']}", axis=1),
        textposition='auto',
        hovertemplate='%{x}<br>Found: %{y}<br>Total: %{text}<extra></extra>'
    ))

    # Add bars for incorrect answers
    fig.add_trace(go.Bar(
        x=incorrect_data[actual_group],
        y=incorrect_data['Found'],
        name='Incorrect Answers',
        marker_color='#F44336',
        text=incorrect_data.apply(lambda x: f"{x['Found']}/{x['Total']}", axis=1),
        textposition='auto',
        hovertemplate='%{x}<br>Found: %{y}<br>Total: %{text}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'Gold Context Presence by Answer Correctness and {group_option}',
        barmode='group',
        xaxis_title=group_option,
        yaxis_title='Count',
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
