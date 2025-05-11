"""
Sidebar view for RAG Configuration
"""
import streamlit as st

def render_sidebar(chunking_strategies):
    """
    Render the sidebar with chunking strategies and configuration options
    
    Returns:
        tuple: (selected_strategies, chunk_size, overlap, process_button_clicked)
    """
    st.sidebar.title("RAG Configuration")

    # Define callbacks for UI elements
    def on_strategy_select():
        pass  # Session state is updated automatically via the key parameter
    
    def on_chunk_size_change():
        # Keep overlap less than chunk_size - 50
        if st.session_state.overlap > st.session_state.chunk_size - 50:
            st.session_state.overlap = max(0, st.session_state.chunk_size - 50)
    
    # Use multiselect with default value from session state
    selected_strategies = st.sidebar.multiselect(
        "Select Chunking Strategies",
        chunking_strategies,
        default=st.session_state.selected_strategies,
        key="selected_strategies",
        on_change=on_strategy_select
    )
    
    # Handle case when no strategy is selected
    if not selected_strategies:
        st.sidebar.warning("Please select at least one chunking strategy.")
        selected_strategies = []
    
    # Only show chunk size slider for strategies that use fixed size chunks
    needs_chunk_size = any(s in selected_strategies for s in [
        "FixedSizeChunkingStrategy", 
        "SlidingWindowChunkingStrategy", 
        "SentenceBasedChunkingStrategy", 
        "ParagraphBasedChunkingStrategy"
    ])
    
    if needs_chunk_size:
        chunk_size = st.sidebar.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=5000,
            value=st.session_state.chunk_size,
            step=100,
            help="Number of characters per chunk",
            key="chunk_size",
            on_change=on_chunk_size_change
        )
    else:
        chunk_size = st.session_state.chunk_size
    
    # Only show overlap slider for SlidingWindowChunkingStrategy
    if "SlidingWindowChunkingStrategy" in selected_strategies:
        # Calculate safe maximum overlap
        max_overlap = max(0, chunk_size - 50)
        # Ensure current overlap value is valid
        safe_overlap_value = min(st.session_state.overlap, max_overlap)
        
        overlap = st.sidebar.slider(
            "Overlap (characters)",
            min_value=0,
            max_value=max_overlap,
            value=safe_overlap_value,
            step=10,
            help="Number of overlapping characters between chunks",
            key="overlap"
        )
    else:
        overlap = st.session_state.overlap
    
    # Determine the button text based on whether documents have been processed before
    process_button_text = "Process Documents Again" if st.session_state.has_processed_once else "Process Documents"
    
    # Add Process Documents button - disabled when processing is in progress
    process_button = st.sidebar.button(
        process_button_text, 
        disabled=st.session_state.is_processing
    )
    
    # Immediately set processing state to True if the button was clicked
    if process_button:
        st.session_state.is_processing = True
        # Return True to indicate button was clicked for rerunning the app
        return selected_strategies, chunk_size, overlap, True
        
    return selected_strategies, chunk_size, overlap, False
