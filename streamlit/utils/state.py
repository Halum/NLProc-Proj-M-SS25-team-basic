"""
Session state management utilities
"""
import streamlit as st

def initialize_session_state(chunking_strategies):
    """
    Initialize all session state variables with defaults
    """
    session_state_defaults = {
        'selected_strategies': chunking_strategies.copy(),
        'is_processing': False,
        'processed_strategies': [],
        'insights_loaded': False,
        'chunk_size': 1000,
        'overlap': 100,
        'selected_query': '',
        'selected_interaction_strategies': [],
        'has_processed_once': False,
        'processing_times': {},
        'strategy_retrievers': {}  # Store retrievers for each strategy
    }
    
    # Initialize any missing session state values with defaults
    for key, default_value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
