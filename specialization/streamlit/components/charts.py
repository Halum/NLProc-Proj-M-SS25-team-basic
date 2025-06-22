"""
Visualization components for RAG performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_answer_correctness(insights_df):
    """
    Create a visualizations for answer correctness.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Calculate correct vs incorrect counts
    correct_counts = insights_df['is_correct'].value_counts().reset_index()
    correct_counts.columns = ['Is Correct', 'Count']
    
    # Replace boolean values with readable labels
    correct_counts['Is Correct'] = correct_counts['Is Correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Create pie chart
    fig = px.pie(
        correct_counts, 
        names='Is Correct', 
        values='Count', 
        color='Is Correct',
        color_discrete_map={'Correct': '#4CAF50', 'Incorrect': '#F44336'},
        title='Answer Correctness Distribution'
    )
    
    # Update layout
    fig.update_layout(
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
    # Create a new dataframe for plotting
    plot_df = pd.DataFrame({
        'Query': insights_df['question'],
        'Average Similarity': insights_df['avg_similarity_score'],
        'Is Correct': insights_df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    })
    
    # Create histogram of similarity scores colored by correctness
    fig = px.histogram(
        plot_df,
        x='Average Similarity',
        color='Is Correct',
        barmode='group',
        nbins=20,
        color_discrete_map={'Correct': '#4CAF50', 'Incorrect': '#F44336'},
        title='Distribution of Similarity Scores by Answer Correctness'
    )
    
    fig.update_layout(
        xaxis_title='Average Similarity Score',
        yaxis_title='Count',
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create scatter plot of similarity scores vs. query
    fig = px.scatter(
        plot_df,
        x='Average Similarity',
        y='Query',
        color='Is Correct',
        color_discrete_map={'Correct': '#4CAF50', 'Incorrect': '#F44336'},
        title='Similarity Scores by Query'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title='Average Similarity Score',
        yaxis_title='Query',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def plot_bert_rouge_scores(insights_df):
    """
    Create visualizations for BERT and ROUGE scores.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Extract BERT scores if available
    bert_scores = []
    for idx, row in insights_df.iterrows():
        if isinstance(row.get('bert_score'), dict):
            bert_scores.append({
                'Query': row['question'],
                'Precision': row['bert_score'].get('bert_precision', 0),
                'Recall': row['bert_score'].get('bert_recall', 0),
                'F1': row['bert_score'].get('bert_f1', 0),
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect'
            })
    
    # Extract ROUGE scores if available
    rouge_scores = []
    for idx, row in insights_df.iterrows():
        if isinstance(row.get('rouge_score'), dict):
            rouge_scores.append({
                'Query': row['question'],
                'ROUGE-1 F1': row['rouge_score'].get('rouge1_fmeasure', 0),
                'ROUGE-2 F1': row['rouge_score'].get('rouge2_fmeasure', 0),
                'ROUGE-L F1': row['rouge_score'].get('rougeL_fmeasure', 0),
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect'
            })
            
    if bert_scores and rouge_scores:
        # Create dataframes
        bert_df = pd.DataFrame(bert_scores)
        rouge_df = pd.DataFrame(rouge_scores)
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('BERT Scores', 'ROUGE Scores'))
        
        # Add BERT score traces
        fig.add_trace(
            go.Bar(
                x=bert_df['Query'],
                y=bert_df['Precision'],
                name='BERT Precision',
                marker_color='#1f77b4',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=bert_df['Query'],
                y=bert_df['Recall'],
                name='BERT Recall',
                marker_color='#ff7f0e',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=bert_df['Query'],
                y=bert_df['F1'],
                name='BERT F1',
                marker_color='#2ca02c',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add ROUGE score traces
        fig.add_trace(
            go.Bar(
                x=rouge_df['Query'],
                y=rouge_df['ROUGE-1 F1'],
                name='ROUGE-1 F1',
                marker_color='#d62728',
                showlegend=True
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=rouge_df['Query'],
                y=rouge_df['ROUGE-2 F1'],
                name='ROUGE-2 F1',
                marker_color='#9467bd',
                showlegend=True
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=rouge_df['Query'],
                y=rouge_df['ROUGE-L F1'],
                name='ROUGE-L F1',
                marker_color='#8c564b',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(tickangle=45, title_text="Query")
        fig.update_yaxes(title_text="Score", range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("BERT and/or ROUGE score data is not available in the insights.")
        
        # If we have some data but not other, display what we have
        if bert_scores:
            bert_df = pd.DataFrame(bert_scores)
            st.subheader("BERT Scores")
            st.dataframe(bert_df[['Query', 'Precision', 'Recall', 'F1', 'Is Correct']])
            
        if rouge_scores:
            rouge_df = pd.DataFrame(rouge_scores)
            st.subheader("ROUGE Scores")
            st.dataframe(rouge_df[['Query', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1', 'Is Correct']])
