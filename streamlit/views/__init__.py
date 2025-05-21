"""
Views package for Streamlit app
"""
from .sidebar import render_sidebar
from .preprocessing_tab import render_processing_ui, process_documents, display_processing_results
from .interaction_tab import render_interaction_ui, process_query
from .insights_tab import render_insights_ui, display_insights
from .chat_tab import render_chat_ui, process_chat_message

__all__ = [
    'render_sidebar',
    'render_processing_ui',
    'process_documents',
    'display_processing_results',
    'render_interaction_ui',
    'process_query',
    'render_insights_ui',
    'display_insights',
    'render_chat_ui',
    'process_chat_message'
]
