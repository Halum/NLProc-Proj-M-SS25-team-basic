"""
Visualization components for RAG performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    
    # Ensure Count column is numeric
    correct_counts['Count'] = pd.to_numeric(correct_counts['Count'])
    
    # Create pie chart directly from DataFrame
    fig = px.pie(
        correct_counts,
        values='Count',
        names='Is Correct',
        color='Is Correct',
        color_discrete_map={'Correct': '#4CAF50', 'Incorrect': '#F44336'},
        title='Answer Correctness Distribution'
    )
    # Make sure we're working with explicit values in case of any type issues
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
    
def plot_bert_scores(insights_df):
    """
    Create visualization for BERT scores.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Extract BERT scores if available
    bert_scores = []
    for idx, row in insights_df.iterrows():
        if isinstance(row.get('bert_score'), dict):
            # Ensure we're explicitly converting to float and handling potential string values
            try:
                precision = float(row['bert_score'].get('bert_precision', 0))
                recall = float(row['bert_score'].get('bert_recall', 0))
                f1 = float(row['bert_score'].get('bert_f1', 0))
            except (ValueError, TypeError):
                # Handle potential conversion issues
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                
            bert_scores.append({
                'Query': row['question'],
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Is Correct': 'Correct' if row['is_correct'] else 'Incorrect'
            })
            
    if bert_scores:
        # Create dataframe
        bert_df = pd.DataFrame(bert_scores)
        
        # Sort by F1 score to make the chart more readable
        bert_df = bert_df.sort_values(by='F1', ascending=True)
        
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
        
        # Update layout
        fig.update_layout(
            title="BERT Scores <br>(Higher scores indicate better semantic matching)",
            height=max(450, len(bert_df) * 30),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=150, l=200),
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
    else:
        st.warning("BERT score data is not available in the insights.")

def plot_rouge_scores(insights_df):
    """
    Create visualization for ROUGE scores.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
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
            
    if rouge_scores:
        # Create dataframe
        rouge_df = pd.DataFrame(rouge_scores)
        
        # Sort by ROUGE-L F1 score to make the chart more readable
        rouge_df = rouge_df.sort_values(by='ROUGE-L F1', ascending=True)

        # Create figure for ROUGE scores
        fig = go.Figure()
        
        # Add ROUGE score traces
        fig.add_trace(
            go.Bar(
                y=rouge_df['Query'].tolist(),
                x=rouge_df['ROUGE-1 F1'].tolist(),
                name='ROUGE-1 F1',
                marker_color='#d62728',
                orientation='h'
            )
        )
        
        fig.add_trace(
            go.Bar(
                y=rouge_df['Query'].tolist(),
                x=rouge_df['ROUGE-2 F1'].tolist(),
                name='ROUGE-2 F1',
                marker_color='#9467bd',
                orientation='h'
            )
        )
        
        fig.add_trace(
            go.Bar(
                y=rouge_df['Query'].tolist(),
                x=rouge_df['ROUGE-L F1'].tolist(),
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
        
        # Update layout
        fig.update_layout(
            title="ROUGE Scores <br>(Higher scores indicate better text matching)",
            height=max(450, len(rouge_df) * 30),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=150, l=200),
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
    else:
        st.warning("ROUGE score data is not available in the insights.")
        if rouge_scores:
            rouge_df = pd.DataFrame(rouge_scores)
            st.subheader("ROUGE Scores")
            st.dataframe(rouge_df[['Query', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1', 'Is Correct']])
