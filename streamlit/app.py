"""
Streamlit app for exploring RAG with different chunking strategies
"""
import sys
import os
import streamlit as st

# Add the parent directory to the sys.path to import from baseline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the baseline directory to the Python path so that internal imports work
baseline_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline")
if baseline_dir not in sys.path:
    sys.path.append(baseline_dir)

# Import views and utils modules
from utils.state import initialize_session_state  # noqa: E402
from views import (  # noqa: E402
    render_sidebar,
    render_processing_ui, 
    process_documents, 
    load_insights_file, 
    display_processing_results,
    render_interaction_ui,
    process_query,
    display_insights
)

def main():
    """
    Main function to set up the Streamlit app layout
    """
    # Set up the app title
    st.set_page_config(
        page_title="RAG Chunking Strategies Explorer",
        layout="wide"
    )

    # Define available chunking strategies
    chunking_strategies = [
        "FixedSizeChunkingStrategy",
        "SlidingWindowChunkingStrategy",
        "SentenceBasedChunkingStrategy", 
        "ParagraphBasedChunkingStrategy",
        "SemanticChunkingStrategy",
        "MarkdownHeaderChunkingStrategy"
    ]
    
    # Initialize session state
    initialize_session_state(chunking_strategies)
    
    # Render sidebar
    selected_strategies, chunk_size, overlap, process_button_clicked = render_sidebar(chunking_strategies)
    
    # Rerun app if process button was clicked
    if process_button_clicked:
        st.rerun()

    # Main content area
    st.title("RAG Chunking Strategies Explorer")
    
    # Create tabs for Preprocessing and Interaction
    preprocessing_tab, interaction_tab = st.tabs(["Preprocessing", "Interaction"])
    
    # Render preprocessing tab
    with preprocessing_tab:
        render_processing_ui(selected_strategies)
    
    # Render interaction tab
    with interaction_tab:
        # Debug information to help trace state issues
        retriever_keys = list(st.session_state.strategy_retrievers.keys()) if 'strategy_retrievers' in st.session_state else []
        
        ask_button, query_data, selected_interaction_strategies, insights_df, has_insights = render_interaction_ui()
        
        # Process query when ask button is clicked
        if ask_button and has_insights and query_data and selected_interaction_strategies:
            # Double check that we have processed strategies available
            if not st.session_state.processed_strategies:
                st.error("No processed strategies found. Please process documents in the Preprocessing tab first.")
            else:
                results_by_strategy = process_query(interaction_tab, query_data, selected_interaction_strategies)
                
                # Display insights charts if we have results and insights data
                if results_by_strategy and insights_df is not None:
                    display_insights(insights_df)
    
    # Process documents if we're in processing state
    if st.session_state.is_processing and not process_button_clicked:
        # We're in processing state but not from a new button click (rerun)
        # This prevents double-processing due to the rerun() call
        results = process_documents(preprocessing_tab, selected_strategies, chunk_size, overlap)
        
        # Load insights file
        insights_df = load_insights_file()
        
        # Display processing results if processing was successful
        if results is not None:
            display_processing_results(preprocessing_tab, results)
            
    # Load insights data for the interaction tab if processing was already done
    if st.session_state.has_processed_once and not insights_df:
        insights_df = load_insights_file()


if __name__ == "__main__":
    main()
