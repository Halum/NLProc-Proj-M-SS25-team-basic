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
from specialization.streamlit.utils.styling import configure_page

# Configure logging
logging.basicConfig(
    level=getattr(logging, "INFO", logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration with consistent styling
configure_page("Historical Analysis", "📊")

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
