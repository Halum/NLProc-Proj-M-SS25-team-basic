"""
Interaction tab view
"""
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baseline.preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)
from baseline.retriever.retreiver import Retriever
from baseline.evaluation.answer_verifier import AnswerVerifier

def render_interaction_ui():
    """
    Render the interaction tab UI with query interfaces
    
    Returns:
        tuple: (ask_button_clicked, query_data, selected_interaction_strategies)
    """
    st.header("Query Interaction")
    st.write("Ask questions about the processed documents and see answers based on different chunking strategies.")
    
    # Try to load insights file first to get data for visualizations
    insights_df = load_insights_data()
    
    # Determine which strategies are available for interaction
    available_strategies, has_insights = get_available_strategies(insights_df)
    
    # Get the sample queries from AnswerVerifier
    sample_queries = [item["query"] for item in AnswerVerifier.get_sample_labeled_data()]
    
    # Create the query selection interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_col, strategy_col = st.columns([2, 2])
        
        selected_query = render_query_selector(query_col, sample_queries)
        selected_interaction_strategies = render_strategy_selector(strategy_col, available_strategies)
    
    with col2:
        # Add the Ask button (aligned with the dropdowns)
        # Add some vertical space to align with dropdowns
        st.write("")
        # Disable during processing or if no strategies selected
        button_disabled = (
            st.session_state.is_processing or 
            not has_insights or 
            not selected_interaction_strategies
        )
        ask_button = st.button(
            "Ask", 
            key="ask_query_button", 
            disabled=button_disabled
        )
    
    # Find the matching query data
    query_data = None
    if selected_query:
        if selected_query == "All":
            # For "All" option, we'll return a special marker
            # The actual processing of all queries will be handled in app.py
            query_data = "ALL_QUERIES"
        else:
            # Find the specific query data
            for item in AnswerVerifier.get_sample_labeled_data():
                if item["query"] == selected_query:
                    query_data = item
                    break
    
    return ask_button, query_data, selected_interaction_strategies, insights_df, has_insights


def load_insights_data():
    """Load insights data for visualization"""
    insights_df = None
    try:
        # Get the directory where insight file should be stored
        insight_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "baseline", "insight")
        
        # Define possible insight file paths
        insight_path = os.path.join(insight_dir, "chunking_strategy_insights.csv")
        alternative_path = os.path.join(insight_dir, "chunking_strategy_insights.csv.csv")
        
        # Try to load the insights file
        if os.path.exists(insight_path):
            insights_df = pd.read_csv(insight_path)
            st.session_state.insights_loaded = True
        elif os.path.exists(alternative_path):
            insights_df = pd.read_csv(alternative_path)
            st.session_state.insights_loaded = True
    except Exception as e:
        st.warning(f"Could not load insights file: {str(e)}")
        st.session_state.insights_loaded = False
    
    return insights_df


def get_available_strategies(insights_df):
    """Determine which strategies are available for interaction"""
    # Get both session-tracked processed strategies and those from insights file
    processed_from_session = st.session_state.processed_strategies
    file_strategies = []
    if insights_df is not None:
        file_strategies = insights_df['chunk_strategy'].unique().tolist()
    
    # Combine both sources to get available strategies
    if processed_from_session or file_strategies:
        # Use strategies from either source
        combined_strategies = list(set(processed_from_session + file_strategies))
        available_strategies = sorted(combined_strategies)
        has_insights = True
        
        # If we're actively processing, update the UI
        if st.session_state.is_processing:
            st.info("Processing documents... Interaction tab will be updated when finished.")
    else:
        # No strategies available yet
        available_strategies = []
        has_insights = False
        st.warning("Please process documents first using the sidebar options.")
    
    return available_strategies, has_insights


def render_query_selector(container, sample_queries):
    """Render the query selection dropdown"""
    with container:
        # Add "All" option to the beginning of the list
        options = ["All"] + sample_queries
        
        # Make sure selected query is initialized with valid value
        if not st.session_state.selected_query:
            st.session_state.selected_query = options[0]  # Default to "All"
        elif st.session_state.selected_query not in options:
            st.session_state.selected_query = options[0]  # Reset to "All" if invalid
        
        # Define a callback that will be triggered when selection changes
        def on_query_change():
            pass  # Session state is updated automatically via key parameter
        
        # Use selectbox with direct link to session state
        selected_query = st.selectbox(
            "Select a Query", 
            options, 
            key="selected_query",  # Direct link to session state
            index=options.index(st.session_state.selected_query) if st.session_state.selected_query in options else 0,
            on_change=on_query_change
        )
        
        return selected_query


def render_strategy_selector(container, available_strategies):
    """Render the strategy selection dropdown"""
    with container:
        # Make sure we have a valid selection if strategies are available
        if available_strategies:
            # If we don't have any selected strategies yet or our selections are no longer valid
            if not st.session_state.selected_interaction_strategies:
                # Default to first available strategy
                st.session_state.selected_interaction_strategies = [available_strategies[0]]
            else:
                # Only keep valid selections
                valid_selections = [s for s in st.session_state.selected_interaction_strategies 
                                    if s in available_strategies]
                
                # If we lost all valid selections, default to first available
                if not valid_selections and available_strategies:
                    valid_selections = [available_strategies[0]]
                    
                st.session_state.selected_interaction_strategies = valid_selections
        
        # Define a callback for interaction strategy selection
        def on_interaction_strategy_select():
            # This approach prevents the double-click issue
            pass  # The session state is updated automatically via the key parameter
        
        # Create a multiselect for strategies (only show used strategies)
        if available_strategies:
            # Show a more helpful message when no strategies are processed
            selected_interaction_strategies = st.multiselect(
                "Select Strategies", 
                available_strategies,
                default=st.session_state.selected_interaction_strategies,
                key="selected_interaction_strategies",  # Direct link to session_state
                on_change=on_interaction_strategy_select
            )
        else:
            # Display message to guide user
            st.info("Process some chunking strategies in the sidebar first")
            selected_interaction_strategies = st.multiselect(
                "Select Strategies", 
                ["No strategies available"], 
                disabled=True,
                key="interaction_strategy_multiselect_disabled"
            )
            
        return selected_interaction_strategies


def process_query(tab, query_data, selected_interaction_strategies):
    """Process the query with selected strategies and display results"""
    with tab:
        st.subheader("Query Results")
        
        results_by_strategy = {}
        
        # Handle "ALL_QUERIES" marker (All queries option)
        if query_data == "ALL_QUERIES":
            # Get all sample queries
            all_sample_queries = AnswerVerifier.get_sample_labeled_data()
            st.info(f"Processing all {len(all_sample_queries)} queries...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process each query
            for i, query_item in enumerate(all_sample_queries):
                with st.expander(f"Query {i+1}: {query_item['query']}", expanded=False):
                    query_results = process_single_query(query_item, selected_interaction_strategies)
                    
                    # Display results for this query - indicate we're inside an expander
                    display_query_results(query_results, query_item, in_expander=True)
                    
                    # Store results for insights analysis
                    for strategy, result in query_results.items():
                        if strategy not in results_by_strategy:
                            results_by_strategy[strategy] = []
                        results_by_strategy[strategy].append(result)
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(all_sample_queries))
            
            # Display a summary
            st.success(f"Processed all {len(all_sample_queries)} queries with {len(selected_interaction_strategies)} strategies")
            
        else:
            # Process a single query
            results_by_strategy = process_single_query(query_data, selected_interaction_strategies)
            
            # Display results if we have any
            if results_by_strategy:
                display_query_results(results_by_strategy, query_data)
        
        return results_by_strategy


def process_single_query(query_data, selected_interaction_strategies, tab=None):
    """Process a single query with selected strategies and display results if tab is provided"""
    results_by_strategy = {}
    container = tab if tab else st
    
    # Process the query for each selected strategy
    for selected_strategy in selected_interaction_strategies:
        with container.spinner(f"Processing query with {selected_strategy}..."):
            try:
                # Use the pre-processed retriever from session state if available
                if 'strategy_retrievers' in st.session_state and selected_strategy in st.session_state.strategy_retrievers:
                    # Use existing retriever that was stored during document processing
                    retriever = st.session_state.strategy_retrievers[selected_strategy]
                    container.info(f"Using pre-processed retriever for {selected_strategy}")
                else:
                    # Fall back to creating a new retriever if not found (should not happen normally)
                    container.warning(f"No pre-processed data found for {selected_strategy}. Creating a new retriever...")
                    
                    # Initialize the strategy
                    if selected_strategy == "FixedSizeChunkingStrategy":
                        strategy = FixedSizeChunkingStrategy(chunk_size=st.session_state.chunk_size)
                    elif selected_strategy == "SlidingWindowChunkingStrategy":
                        strategy = SlidingWindowChunkingStrategy(chunk_size=st.session_state.chunk_size, overlap=st.session_state.overlap)
                    elif selected_strategy == "SentenceBasedChunkingStrategy":
                        strategy = SentenceBasedChunkingStrategy(chunk_size=st.session_state.chunk_size)
                    elif selected_strategy == "ParagraphBasedChunkingStrategy":
                        strategy = ParagraphBasedChunkingStrategy(chunk_size=st.session_state.chunk_size)
                    elif selected_strategy == "SemanticChunkingStrategy":
                        strategy = SemanticChunkingStrategy()
                    elif selected_strategy == "MarkdownHeaderChunkingStrategy":
                        strategy = MarkdownHeaderChunkingStrategy()
                    
                    # Create a new retriever and process documents
                    retriever = Retriever(strategy)
                    from baseline.config.config import DOCUMENT_FOLDER_PATH
                    retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)
                    retriever.preprocess()
                    retriever.save()
                
                # Load the processed chunks
                retrieved_chunks, distances = retriever.load(query_data["query"])
                
                # Generate answer
                generated_answer = retriever.query(query_data["query"], retrieved_chunks)
                
                # Find if the context is in retrieved chunks
                expected_chunk_index, expected_chunk = AnswerVerifier.find_chunk_containing_context(retrieved_chunks, context=query_data["context"])
                
                # Store results for this strategy
                results_by_strategy[selected_strategy] = {
                    "generated_answer": generated_answer,
                    "retrieved_chunks": retrieved_chunks,
                    "expected_chunk_index": expected_chunk_index,
                    "expected_chunk": expected_chunk,
                    "distances": distances
                }
                
            except Exception as e:
                container.error(f"Error processing query with {selected_strategy}: {str(e)}")
    
    return results_by_strategy


def display_query_results(results_by_strategy, query_data, in_expander=False):
    """Display the query results in tabs for each strategy
    
    Args:
        results_by_strategy: Dictionary mapping strategy names to their results
        query_data: Dictionary containing query, answer, and context
        in_expander: Boolean indicating if we're already inside an expander
    """
    # Get the expected answer and context
    expected_answer = query_data["answer"]
    context = query_data["context"]
    
    # Check if we have a single result or multiple results
    if not results_by_strategy:
        st.warning("No results to display")
        return
        
    # Create tabs for each strategy
    if len(results_by_strategy) > 1:
        # Multiple strategies - use tabs
        strategy_tabs = st.tabs(list(results_by_strategy.keys()))
        
        # Display results in tabs
        for i, (strategy_name, results) in enumerate(results_by_strategy.items()):
            with strategy_tabs[i]:
                results_col1, results_col2 = st.columns(2)
                
                with results_col1:
                    st.markdown("**Generated Answer:**")
                    st.write(results["generated_answer"])
                
                with results_col2:
                    st.markdown("**Expected Answer:**")
                    st.write(expected_answer)
                
                st.markdown("**Context:**")
                st.write(context)
                
                if results["expected_chunk_index"] != -1:
                    st.success(f"Context found in chunk #{results['expected_chunk_index']+1}")
                else:
                    st.error("Context not found in retrieved chunks")
                
                # Show retrieved chunks based on whether we're in an expander
                if in_expander:
                    st.markdown("**Retrieved Chunks:**")
                    chunks_container = st.container()
                    with chunks_container:
                        for i, chunk in enumerate(results["retrieved_chunks"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            # Use a container with CSS styling to wrap text
                            st.markdown(
                                f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                                unsafe_allow_html=True
                            )
                            st.markdown("---")
                else:
                    with st.expander("Retrieved Chunks", expanded=False):
                        for i, chunk in enumerate(results["retrieved_chunks"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            # Use a container with CSS styling to wrap text
                            st.markdown(
                                f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                                unsafe_allow_html=True
                            )
                            st.markdown("---")
    else:
        # Just one strategy - no need for tabs
        strategy_name = list(results_by_strategy.keys())[0]
        results = results_by_strategy[strategy_name]
        
        st.subheader(f"Results using {strategy_name}")
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.markdown("**Generated Answer:**")
            st.write(results["generated_answer"])
        
        with results_col2:
            st.markdown("**Expected Answer:**")
            st.write(expected_answer)
        
        st.markdown("**Context:**")
        st.write(context)
        
        if results["expected_chunk_index"] != -1:
            st.success(f"Context found in chunk #{results['expected_chunk_index']+1}")
        else:
            st.error("Context not found in retrieved chunks")
        
        # Show retrieved chunks based on whether we're in an expander
        if in_expander:
            st.markdown("**Retrieved Chunks:**")
            chunks_container = st.container()
            with chunks_container:
                for i, chunk in enumerate(results["retrieved_chunks"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    # Use a container with CSS styling to wrap text
                    st.markdown(
                        f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
        else:
            # Regular case - use expander
            with st.expander("Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(results["retrieved_chunks"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    # Use a container with CSS styling to wrap text
                    st.markdown(
                        f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                        unsafe_allow_html=True
                    )
                    st.markdown("---")


def display_insights(insights_df):
    """Display strategy insights visualizations"""
    if insights_df is not None:
        st.subheader("Chunking Strategy Insights")
        
        # Filter insights by strategy if selected
        filtered_df = insights_df
        
        # Create visual insights with smaller graphs side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Number of chunks by strategy
            st.markdown("### Number of Chunks")
            chunks_by_strategy = filtered_df.groupby('chunk_strategy')['number_of_chunks'].mean().reset_index()
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.barplot(data=chunks_by_strategy, x='chunk_strategy', y='number_of_chunks', ax=ax1)
            ax1.set_xlabel('Strategy')
            ax1.set_ylabel('Number of Chunks')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # 2. Correct Answer Rate by Strategy
            st.markdown("### Correct Answer Rate")
            correct_answer_rate = filtered_df.groupby('chunk_strategy')['correct_answer'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=correct_answer_rate, x='chunk_strategy', y='correct_answer', ax=ax2)
            
            # Add percentage labels
            for i, bar in enumerate(bars.patches):
                bars.text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + 0.01, 
                        f'{bar.get_height():.0%}',
                        ha='center', va='bottom')
                        
            ax2.set_xlabel('Strategy')
            ax2.set_ylabel('Correct Answer Rate')
            ax2.set_ylim(0, 1.1)  # Set y-axis limit to accommodate percentage labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 3. Context Found Rate by Strategy
            st.markdown("### Context Found Rate")
            context_found_df = filtered_df.copy()
            context_found_df['context_found'] = context_found_df['retrieved_chunk_rank'] != -1
            context_found_rate = context_found_df.groupby('chunk_strategy')['context_found'].mean().reset_index()
            
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            bars = sns.barplot(data=context_found_rate, x='chunk_strategy', y='context_found', ax=ax3)
            
            # Add percentage labels
            for i, bar in enumerate(bars.patches):
                bars.text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + 0.01, 
                        f'{bar.get_height():.0%}',
                        ha='center', va='bottom')
                        
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('Context Found Rate')
            ax3.set_ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3)
        
        # Show raw data in the other column
        with col4:
            st.markdown("### Raw Insights Data")
            st.dataframe(filtered_df, height=300)
