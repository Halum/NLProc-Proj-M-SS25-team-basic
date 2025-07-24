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
from specialization.streamlit.utils.data_loader import get_sorted_iteration_notes
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

# Display iteration notes from result_interpretation.json
try:
    with st.spinner("Loading iteration notes..."):
        iteration_notes = get_sorted_iteration_notes()
        
        if iteration_notes:
            st.subheader("Iteration History Notes")
            
            with st.expander("View Iteration Notes", expanded=True):
                # Create a formatted table for iteration notes
                st.markdown("### Model & Data Iteration History")
                st.markdown("This table shows changes made across different evaluation iterations, providing context for the performance trends.")
                
                # Display iteration notes in a more compact format
                # Counter to track iteration number
                iteration_count = 1
                
                for iteration_id, note, timestamp in iteration_notes:
                    formatted_date = timestamp.strftime("%Y-%m-%d")
                    if timestamp.year != 1900:  # Check if we have a valid timestamp
                        display_text = f"**{formatted_date} (Iteration {iteration_count}):** {note}"
                    else:
                        display_text = f"**{iteration_id} (Iteration {iteration_count}):** {note}"
                        
                    st.markdown(display_text)
                    iteration_count += 1
                
                st.markdown("---")
        else:
            st.info("No iteration notes available.")
except Exception as e:
    st.error(f"Error loading iteration notes: {str(e)}")

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
