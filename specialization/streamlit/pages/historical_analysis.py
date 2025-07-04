"""
Historical Analysis Page.

This page provides visualizations of historical performance metrics across multiple evaluation runs.
It shows trends in BERT scores, ROUGE scores, and similarity scores over time.
"""

import streamlit as st
import logging

# Add the project root to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import utilities for the dashboard
from specialization.streamlit.utils.historical_data import get_historical_metrics
from specialization.streamlit.views.historical_charts import display_historical_charts

# Configure logging
logging.basicConfig(
    level=getattr(logging, "INFO", logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title="Historical Analysis",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for consistent styling
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

# Title and description
st.title("📈 Historical Performance Trends")
st.markdown("""
This page visualizes how RAG system performance metrics have changed over time, allowing 
you to track improvements and regressions across multiple evaluation runs.
""")

# No configuration section displayed in the UI anymore

# Fetch historical metrics data
try:
    with st.spinner("Loading historical metrics data..."):
        historical_data = get_historical_metrics()

        if historical_data is not None and not historical_data.empty:
            # Show number of runs available
            st.success(f"Loaded data from {len(historical_data)} evaluation runs")
            
            try:
                # Display historical charts
                display_historical_charts(historical_data)
            except Exception as e:
                st.error(f"Error displaying charts: {str(e)}")
        else:
            st.error("No historical metrics data available. Please ensure you have multiple timestamped insight files.")
            
            st.info("""
            Historical analysis requires:
            1. Multiple evaluation runs with timestamped output files
            2. Each file must contain BERT scores, ROUGE scores, and similarity metrics
            """)
except Exception as e:
    st.error(f"Error loading historical metrics: {str(e)}")
