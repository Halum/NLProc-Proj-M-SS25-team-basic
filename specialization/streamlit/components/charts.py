"""
Visualization components for RAG performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statistics

def calculate_dynamic_height(df, query_col='Query', min_height=600, base_multiplier=40, text_factor=0.8):
    """
    Calculate dynamic chart height based on number of entries and query text length.
    
    Args:
        df (pd.DataFrame): DataFrame containing queries
        query_col (str): Name of the column containing query text
        min_height (int): Minimum height in pixels
        base_multiplier (int): Base pixels per row
        text_factor (float): Factor for additional height based on text length
    
    Returns:
        int: Calculated height in pixels
    """
    # Get number of entries
    num_entries = len(df)
    
    # Calculate average query text length if the query column exists
    if query_col in df.columns:
        text_lengths = [len(str(q)) for q in df[query_col]]
        if text_lengths:
            avg_length = statistics.mean(text_lengths)
            # Add height for longer than average queries
            text_adjustment = max(0, avg_length - 50) * text_factor
        else:
            text_adjustment = 0
    else:
        text_adjustment = 0
    
    # Calculate height with minimum value
    height = max(min_height, int(num_entries * (base_multiplier + text_adjustment)))
    
    return height

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
    
    # Make sure we have numeric values for similarity scores
    plot_df['Average Similarity'] = pd.to_numeric(plot_df['Average Similarity'], errors='coerce')
    
    # Drop any NaN values after conversion
    plot_df = plot_df.dropna(subset=['Average Similarity'])
    
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
    chart_height = calculate_dynamic_height(plot_df)
    
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
        chart_height = calculate_dynamic_height(bert_df)
        
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
        
        # Calculate dynamic height based on number of entries and query text length
        chart_height = calculate_dynamic_height(rouge_df)
        
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
    else:
        st.warning("ROUGE score data is not available in the insights.")
        if rouge_scores:
            rouge_df = pd.DataFrame(rouge_scores)
            st.subheader("ROUGE Scores")
            st.dataframe(rouge_df[['Query', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1', 'Is Correct']])

def plot_gold_context_presence(insights_df):
    """
    Create visualization showing if gold_context is present in the retrieved context list.
    
    Args:
        insights_df (pd.DataFrame): DataFrame containing evaluation insights
    """
    # Create a new list for tracking presence of gold context in retrieved contexts
    gold_context_presence = []
    
    for idx, row in insights_df.iterrows():
        # Check if both gold_context and context exist in the row
        if 'gold_context' in row and 'context' in row:
            gold_context = row['gold_context']
            retrieved_contexts = row['context']
            
            # Initialize as not found
            is_found = False
            position = -1
            
            # We need to handle different formats of gold_context
            gold_content = None
            
            # Extract content from gold context based on its structure
            if isinstance(gold_context, list):
                # If gold_context is a list, join all contents
                gold_content = " ".join([
                    ctx.get('content', '') if isinstance(ctx, dict) else str(ctx)
                    for ctx in gold_context
                ])
            elif isinstance(gold_context, dict):
                # If gold_context is a dict, get the content field
                gold_content = gold_context.get('content', '')
            else:
                # If gold_context is a string or other type
                gold_content = str(gold_context)
                
            # Check if gold context is present in any of the retrieved contexts
            if isinstance(retrieved_contexts, list):
                for i, ctx in enumerate(retrieved_contexts):  # Check all retrieved contexts
                    if isinstance(ctx, dict) and 'content' in ctx:
                        # Compare content
                        if gold_content and gold_content in ctx['content']:
                            is_found = True
                            position = i
                            break
                    elif isinstance(ctx, str):
                        # Direct string comparison
                        if gold_content and gold_content in ctx:
                            is_found = True
                            position = i
                            break
            
            # Add the result to our list
            gold_context_presence.append({
                'Query': row['question'],
                'Gold Context Present': is_found,
                'Position': position + 1 if position >= 0 else "Not found",  # Convert to 1-indexed
                'Is Correct': row['is_correct']
            })
    
    if gold_context_presence:
        # Create dataframe for visualization
        presence_df = pd.DataFrame(gold_context_presence)
        
        # Compute statistics
        total_queries = len(presence_df)
        found_count = sum(presence_df['Gold Context Present'])
        found_percentage = found_count / total_queries * 100 if total_queries > 0 else 0
        
        # Create two separate visualizations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Create pie chart of gold context presence
            fig_pie = go.Figure()
            fig_pie.add_trace(
                go.Pie(
                    labels=['Present', 'Not Present'],
                    values=[found_count, total_queries - found_count],
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
            # Create a horizontal bar chart with position information
            position_counts = presence_df[presence_df['Gold Context Present']].groupby('Position').size().reset_index(name='Count')
            position_counts = position_counts.sort_values('Position')
            
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    y=position_counts['Position'].astype(str),
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
        
        # Update layout
        # The layout is now handled separately for each figure (fig_pie and fig_bar)
        
        # Add a breakdown by correctness
        st.subheader("Relationship with Answer Correctness")
        
        # Calculate presence rates for correct vs incorrect answers
        correct_df = presence_df[presence_df['Is Correct']]
        incorrect_df = presence_df[~presence_df['Is Correct']]
        
        # Only proceed if we have both correct and incorrect answers
        if not correct_df.empty and not incorrect_df.empty:
            correct_presence = correct_df['Gold Context Present'].mean() * 100
            incorrect_presence = incorrect_df['Gold Context Present'].mean() * 100
            
            # Create a grouped bar chart
            fig_corr = go.Figure()
            fig_corr.add_trace(
                go.Bar(
                    x=['Correct Answers', 'Incorrect Answers'],
                    y=[correct_presence, incorrect_presence],
                    marker_color=['#4CAF50', '#F44336'],
                    text=[f"{correct_presence:.1f}%", f"{incorrect_presence:.1f}%"],
                    textposition='auto'
                )
            )
            
            fig_corr.update_layout(
                title="Gold Context Presence by Answer Correctness",
                yaxis_title="Gold Context Present (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient data to compare gold context presence between correct and incorrect answers.")
    else:
        st.warning("Gold context presence data could not be calculated. Please ensure gold_context and context fields are available in the insights data.")
