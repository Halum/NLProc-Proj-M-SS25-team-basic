"""
Streamlit styling utilities for consistent UI across all pages.

This module provides reusable CSS styling functions to ensure
consistent sidebar width and layout across all app pages.
"""

import streamlit as st


def apply_consistent_page_styling():
    """
    Apply consistent CSS styling across all Streamlit pages.
    
    This function should be called after st.set_page_config() 
    on every page to ensure consistent:
    - Sidebar width (200px)
    - Horizontal block spacing
    - Title case navigation text
    """
    st.markdown("""
    <style>
        /* Reduce sidebar width */
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 200px !important;
        }
        /* Add extra spacing between horizontal blocks */
        [data-testid="stHorizontalBlock"] {
            gap: 3rem !important;
        }
        [data-testid="stHorizontalBlock"] > div:first-child {
            margin-right: 4rem;
            padding-right: 2rem;
        }
        /* Fix sidebar navigation text to be title case */
        section[data-testid="stSidebarUserContent"] .css-17lntkn {
            text-transform: capitalize !important;
        }
        section[data-testid="stSidebarUserContent"] .css-17lntkn:first-letter {
            text-transform: uppercase !important;
        }
    </style>
    """, unsafe_allow_html=True)


def apply_main_page_styling():
    """
    Apply styling specific to the main app page.
    
    This includes the consistent page styling plus any additional
    styling specific to the main dashboard page.
    """
    apply_consistent_page_styling()
    # Add any main-page specific styling here if needed


def configure_page(title: str, icon: str = "📊", layout: str = "wide") -> None:
    """
    Configure a Streamlit page with consistent settings and styling.
    
    Args:
        title: The page title
        icon: The page icon (default: "📊")
        layout: The page layout (default: "wide")
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout
    )
    apply_consistent_page_styling()
