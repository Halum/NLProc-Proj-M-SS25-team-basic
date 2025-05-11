"""
Views package for Streamlit app
"""
from views.sidebar import render_sidebar
from views.preprocessing_tab import render_processing_ui, process_documents, display_processing_results
from views.interaction_tab import render_interaction_ui, process_query
from views.insights_tab import render_insights_ui, display_insights

__all__ = [
    'render_sidebar',
    'render_processing_ui',
    'process_documents',
    'display_processing_results',
    'render_interaction_ui',
    'process_query',
    'render_insights_ui',
    'display_insights'
]
