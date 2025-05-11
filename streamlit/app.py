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
    display_processing_results,
    render_interaction_ui,
    process_query,
    render_insights_ui
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
    
    # Create tabs for Preprocessing, Interaction, and Insights
    preprocessing_tab, interaction_tab, insights_tab = st.tabs(["Preprocessing", "Interaction", "Insights"])
    
    # Render preprocessing tab
    with preprocessing_tab:
        render_processing_ui(selected_strategies)
    
    # Render interaction tab
    with interaction_tab:
        ask_button, query_data, selected_interaction_strategies, _, has_insights = render_interaction_ui()
        
        # If we've processed documents before, make sure we have consistent state
        if st.session_state.has_processed_once and not has_insights:
            has_insights = bool(st.session_state.processed_strategies)
        
        # Process query when ask button is clicked
        if ask_button and has_insights and query_data and selected_interaction_strategies:
            # Double check that we have processed strategies available
            if not st.session_state.processed_strategies:
                st.error("No processed strategies found. Please process documents in the Preprocessing tab first.")
            else:
                process_query(interaction_tab, query_data, selected_interaction_strategies)
    
    # Render insights tab
    with insights_tab:
        render_insights_ui()
    
    # Process documents if we're in processing state
    if st.session_state.is_processing and not process_button_clicked:
        # We're in processing state but not from a new button click (rerun)
        # This prevents double-processing due to the rerun() call
        results = process_documents(preprocessing_tab, selected_strategies, chunk_size, overlap)
        
        # Display processing results if processing was successful
        if results is not None:
            display_processing_results(preprocessing_tab, results)


if __name__ == "__main__":
    main()
