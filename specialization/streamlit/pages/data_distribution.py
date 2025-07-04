"""
Movie Data Distribution Visualization Page

This page shows the distribution of the sample dataset within the full movie dataset
using dimensionality reduction techniques to create 2D and 3D visualizations.
"""

import streamlit as st
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import visualization components
from specialization.streamlit.views.data_distribution import (
    render_cluster_visualization,
    display_genre_distribution,
    display_dimension_reduction_explanation
)
from specialization.streamlit.utils.dimension_reduction import prepare_cluster_visualization_data

# Page configuration
st.set_page_config(
    page_title="Data Distribution Analysis",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main function to render the data distribution visualization page."""
    # Page header
    st.title("📊 Movie Data Distribution Analysis")
    st.markdown("""
    This page visualizes how the sample dataset is distributed within the full movie dataset.
    The visualizations use dimensionality reduction techniques to project high-dimensional movie features into 2D/3D space.
    """)

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        method = st.selectbox(
            "Dimension Reduction Method",
            options=["tsne", "pca", "umap"],
            format_func=lambda x: x.upper(),
            help="Select the dimensionality reduction algorithm to use"
        )
    
    with col2:
        dimensions = st.radio(
            "Dimensions",
            options=[2, 3],
            index=1,
            help="Choose between 2D or 3D visualization",
            horizontal=True
        )
        
    with col3:
        cache_option = st.checkbox(
            "Use cached data", 
            value=True,
            help="Uncheck to recalculate dimension reduction (may be slow)"
        )

    # Main visualization
    st.header("Cluster Visualization")
    st.info("Red points represent the sample dataset, while grey points represent the full dataset.")
    
    # Try rendering the visualization with error handling
    try:
        # Render the main visualization
        render_cluster_visualization(
            method=method,
            dimensions=dimensions,
            height=700
        )
    except ImportError as e:
        st.error(f"Error: {str(e)}")
        st.warning("Please install missing packages by running: `pip install umap-learn scikit-learn plotly`")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Try a different dimension reduction method or recalculating with fresh data")
    
    # Additional analysis section
    st.header("Genre Distribution Analysis")
    
    try:
        # Load data (reusing cached data)
        viz_data = prepare_cluster_visualization_data(
            n_components=2,  # 2D is sufficient for the genre analysis
            method='pca',    # PCA is more reliable and doesn't require additional packages
            cache=cache_option,
            random_state=42
        )
        
        # Display genre distribution
        display_genre_distribution(viz_data)
    except Exception as e:
        st.error(f"Could not generate genre distribution: {str(e)}")
    
    # Explanation
    st.header("Understanding the Visualization")
    display_dimension_reduction_explanation()
    
    # Data source information
    st.subheader("About the Dataset")
    st.markdown("""
    The visualization compares:
    
    - **Full Dataset**: The complete movie dataset from `processed_movies_data.json`
    - **Sample Dataset**: The filtered sample used for the RAG system from `processed_movies_data_sample.json`
    
    The sample is created by filtering movies from selected genres and taking a limited number of entries.
    This visualization helps understand how representative the sample is of the full dataset.
    """)
    
    # Footer
    st.markdown("---")
    st.caption("Data source: Processed movie dataset from the Movies Database")

if __name__ == "__main__":
    main()
