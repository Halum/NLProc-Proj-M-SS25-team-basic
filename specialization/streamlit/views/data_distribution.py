"""
3D and 2D visualizations of movie data clusters.

This module contains functions for rendering movie data clusters
in 2D and 3D space, highlighting the sample data within the full dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import logging

# Add the project root to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import data transformation utilities
from specialization.streamlit.utils.dimension_reduction import prepare_cluster_visualization_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_component_description(method: str, component_num: int) -> str:
    """
    Get a description of what each component represents for different methods.
    
    Args:
        method: The dimensionality reduction method ('tsne', 'pca', 'umap')
        component_num: The component number (1, 2, 3, etc.)
    
    Returns:
        A brief description of what the component represents
    """
    if method.lower() == 'pca':
        descriptions = {
            1: "Max variance direction",
            2: "2nd max variance direction", 
            3: "3rd max variance direction"
        }
        return descriptions.get(component_num, f"Component {component_num}")
    elif method.lower() == 'tsne':
        return "Non-linear embedding axis"
    elif method.lower() == 'umap':
        return "Manifold embedding axis"
    else:
        return f"Dimension {component_num}"

def render_cluster_visualization(
    method: str = "tsne", 
    dimensions: int = 3,
    height: int = 800,
    width: Optional[int] = None
) -> None:
    """
    Render a cluster visualization of movie data.
    
    Args:
        method: Dimension reduction method ('tsne', 'pca', or 'umap')
        dimensions: Number of dimensions (2 or 3)
        height: Height of the visualization in pixels
        width: Width of the visualization in pixels (None for auto-width)
    """
    # Load the prepared data
    with st.spinner(f"Preparing {dimensions}D {method.upper()} visualization..."):
        try:
            viz_data = prepare_cluster_visualization_data(
                n_components=dimensions,
                method=method,
                random_state=42,
                cache=True  # Use caching to avoid recomputing
            )
        except Exception as e:
            st.error(f"Failed to prepare visualization data: {e}")
            st.info("If the error persists, try installing additional packages with: pip install umap-learn scikit-learn plotly")
            return
    
    # Extract data
    full_points = np.array(viz_data["full_points"])
    sample_points = np.array(viz_data["sample_points"])
    full_metadata = viz_data["full_metadata"]
    sample_metadata = viz_data["sample_metadata"]
    
    # Create tooltips
    full_hover_texts = [
        f"Title: {meta['title']}<br>" +
        f"ID: {meta['id']}<br>" +
        f"Year: {meta['year']}<br>" +
        f"Genres: {', '.join(meta['genres']) if isinstance(meta['genres'], list) else meta['genres']}"
        for meta in full_metadata
    ]
    
    sample_hover_texts = [
        f"Title: {meta['title']}<br>" +
        f"ID: {meta['id']}<br>" +
        f"Year: {meta['year']}<br>" +
        f"Genres: {', '.join(meta['genres']) if isinstance(meta['genres'], list) else meta['genres']}"
        for meta in sample_metadata
    ]
    
    # Prepare figure
    if dimensions == 3:
        fig = go.Figure()
        
        # Add full dataset points
        fig.add_trace(go.Scatter3d(
            x=full_points[:, 0],
            y=full_points[:, 1],
            z=full_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.4,
                color="#7194C2",  # Light steel blue - more visible than gray
            ),
            name='Full Dataset',
            text=full_hover_texts,
            hoverinfo='text'
        ))
        
        # Add sample dataset points
        fig.add_trace(go.Scatter3d(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            z=sample_points[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
                color='red',
            ),
            name='Sample Dataset',
            text=sample_hover_texts,
            hoverinfo='text'
        ))
        
        # Layout
        fig.update_layout(
            title=f"{method.upper()} 3D Visualization of Movie Dataset",
            scene=dict(
                xaxis_title=f"{method.upper()} Component 1 ({get_component_description(method, 1)})",
                yaxis_title=f"{method.upper()} Component 2 ({get_component_description(method, 2)})",
                zaxis_title=f"{method.upper()} Component 3 ({get_component_description(method, 3)})",
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=height,
            width=width,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
    else:  # 2D visualization
        fig = go.Figure()
        
        # Add full dataset points
        fig.add_trace(go.Scatter(
            x=full_points[:, 0],
            y=full_points[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                opacity=0.5,
                color="#7194C2",  # Light steel blue - more visible than gray
            ),
            name='Full Dataset',
            text=full_hover_texts,
            hoverinfo='text'
        ))
        
        # Add sample dataset points
        fig.add_trace(go.Scatter(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            mode='markers',
            marker=dict(
                size=7,
                opacity=0.8,
                color='red',
            ),
            name='Sample Dataset',
            text=sample_hover_texts,
            hoverinfo='text'
        ))
        
        # Layout
        fig.update_layout(
            title=f"{method.upper()} 2D Visualization of Movie Dataset",
            xaxis_title=f"{method.upper()} Component 1 ({get_component_description(method, 1)})",
            yaxis_title=f"{method.upper()} Component 2 ({get_component_description(method, 2)})",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=height,
            width=width,
            hovermode='closest'
        )
    
    # Display the figure with proper container settings
    if dimensions == 3:
        # For 3D plots, use a centered container with padding to avoid scroll conflicts
        col1, col2, col3 = st.columns([2, 6, 2])  # 1:8:1 ratio for padding
        with col2:
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 2D plots can use full width safely
        st.plotly_chart(fig, use_container_width=True)
    
    # Add information about the visualization
    method_explanations = {
        'pca': 'Principal Component Analysis - finds directions of maximum variance in the data',
        'tsne': 't-SNE - preserves local neighborhoods, good for visualizing clusters',
        'umap': 'UMAP - preserves both local and global structure, faster than t-SNE'
    }
    
    st.write(f"""
    ### Visualization Details
    - **Method**: {method.upper()} ({dimensions}D) - {method_explanations.get(method.lower(), 'Dimensionality reduction method')}
    - **Total movies**: {viz_data['full_size']}
    - **Sample size**: {viz_data['actual_sample_size']} ({viz_data['actual_sample_size']/viz_data['full_size']*100:.1f}% of total)
    - **Target genres**: {', '.join(viz_data['target_genres'])}
    
    **Component Meanings:**
    - Each axis represents a dimension that captures patterns in movie features
    - {get_component_description(method, 1)} (Component 1)
    - {get_component_description(method, 2)} (Component 2)
    {'- ' + get_component_description(method, 3) + ' (Component 3)' if dimensions == 3 else ''}
    """)
    
    # Add detailed explanation of how 3D plotting works
    with st.expander("🔍 How Are Movie Features Transformed to 3D Points?", expanded=False):
        st.write(f"""
        ### From Movie Attributes to {dimensions}D Coordinates
        
        Each movie in our dataset is transformed from **21 numerical features** to {dimensions}D coordinates.
        
        #### **Exact Features Used (21 total):**
        
        **📊 Numeric Features (7):**
        - `budget` - Movie production budget
        - `revenue` - Box office revenue  
        - `popularity` - TMDb popularity score
        - `vote_average` - Average user rating
        - `vote_count` - Number of votes
        - `runtime` - Movie duration in minutes
        - `release_year` - Year of release
        
        **🎭 Genre Features (13 binary indicators):**
        - `genre_Drama`, `genre_Action`, `genre_Comedy`, `genre_Thriller`
        - `genre_Romance`, `genre_Adventure`, `genre_Crime`, `genre_Fantasy`
        - `genre_Mystery`, `genre_Science Fiction`, `genre_History`, etc.
        - Each movie gets 1 or 0 for each genre (can have multiple genres)
        
        **🆔 Identifier:**
        - `id` - Movie identifier
        
        #### **Feature Processing Pipeline:**
        
        **Step 1: Data Cleaning**
        - Handle missing values (filled with median for numeric features)
        - Convert genres from text lists to binary indicators
        - Extract year from release dates
        - Remove outliers and invalid entries
        
        **Step 2: Standardization**
        - All 21 features scaled to mean=0, standard deviation=1
        - Ensures no single feature dominates (e.g., budget vs. runtime)
        - Each movie becomes a point in 21-dimensional space
        
        **Step 3: Dimensionality Reduction ({method.upper()})**
        - Takes 21D vectors → {dimensions}D coordinates
        - Preserves relationships between movies
        - Similar movies stay close together
        
        #### **What Each {dimensions}D Point Represents:**
        
        **🎬 Position Meaning:**
        - **X-axis ({get_component_description(method, 1)})**: Primary pattern in movie features
        - **Y-axis ({get_component_description(method, 2)})**: Secondary pattern in movie features
        {'- **Z-axis (' + get_component_description(method, 3) + ')**: Tertiary pattern in movie features' if dimensions == 3 else ''}
        
        **🎯 Distance = Similarity:**
        - Movies close together → Similar budget, genres, ratings, etc.
        - Movies far apart → Very different characteristics
        
        #### **Real Examples of Clustering:**
        
        **High-Budget Action Movies:**
        - High budget + revenue + Action genre → Cluster together
        - Examples: Marvel movies, blockbusters
        
        **Family Animation Films:**
        - Family + Animation genres + moderate budget → Separate cluster
        - Examples: Pixar, Disney animations
        
        **Independent Dramas:**
        - Low budget + Drama genre + high critical scores → Another cluster
        - Examples: Art house films, indie movies
        
        **Western Mysteries (Sample Target):**
        - Western + Mystery genres → Positioned between Western and Mystery clusters
        - Shows how multi-genre movies bridge different categories
        
        #### **Why This Visualization is Useful:**
        
        🔍 **Sample Analysis**: Red points show how movies with target genres (Family, Mystery, Western) 
        are distributed across the entire movie landscape - not isolated but integrated!
        
        🎨 **Pattern Recognition**: Clusters reveal natural groupings based on budget, genre, popularity, and ratings combined.
        """)
    
    
    # Add color legend explanation
    st.info("""
    **🎨 Color Legend:**
    - 🔵 **Blue**: Full dataset (all movies)
    - 🔴 **Red**: Sample dataset (movies with target genres: Family, Mystery, Western)
    
    The visualization shows how the sample movies are distributed within the context of the entire movie dataset.
    """)

def display_genre_distribution(viz_data: Dict) -> None:
    """
    Display a bar chart comparing genre distribution between full and sample datasets.
    
    Args:
        viz_data: Visualization data from prepare_cluster_visualization_data
    """
    # Extract and count movies by genre (not total genre occurrences)
    def count_movies_by_genre(metadata_list):
        genre_movie_count = {}
        total_movies = len(metadata_list)
        
        for meta in metadata_list:
            genres = meta["genres"]
            if isinstance(genres, str) and genres.startswith('['):
                try:
                    genres = eval(genres)
                except Exception:
                    genres = []
            elif not isinstance(genres, list):
                genres = [genres] if genres else []
            
            # Count each movie once per genre it belongs to
            for genre in genres:
                if genre not in genre_movie_count:
                    genre_movie_count[genre] = 0
                genre_movie_count[genre] += 1
        
        return genre_movie_count, total_movies
    
    # Count for both datasets
    full_genre_counts, full_total = count_movies_by_genre(viz_data["full_metadata"])
    sample_genre_counts, sample_total = count_movies_by_genre(viz_data["sample_metadata"])
    
    # Get all genres present in either dataset and sort by full dataset frequency
    all_genres = set(full_genre_counts.keys()) | set(sample_genre_counts.keys())
    sorted_genres = sorted(all_genres, key=lambda x: full_genre_counts.get(x, 0), reverse=True)
    
    # Take top 15 genres from full dataset
    top_genres = sorted_genres[:15]
    
    # Create dataframe for plotting
    genre_data = []
    target_genres = set(viz_data['target_genres'])
    
    for genre in top_genres:
        full_count = full_genre_counts.get(genre, 0)
        sample_count = sample_genre_counts.get(genre, 0)
        
        # Calculate percentage of movies that have this genre
        full_pct = (full_count / full_total) * 100 if full_total > 0 else 0
        sample_pct = (sample_count / sample_total) * 100 if sample_total > 0 else 0
        
        # Mark if this is a target genre
        genre_name = f"{genre} ⭐" if genre in target_genres else genre
        
        genre_data.append({
            "Genre": genre_name,
            "Dataset": "Full Dataset",
            "Count": full_count,
            "Percentage": full_pct,
            "IsTarget": genre in target_genres
        })
        genre_data.append({
            "Genre": genre_name,
            "Dataset": "Sample Dataset", 
            "Count": sample_count,
            "Percentage": sample_pct,
            "IsTarget": genre in target_genres
        })
    
    df_genre = pd.DataFrame(genre_data)
    
    # Create plot with improved colors
    fig = px.bar(
        df_genre, 
        x="Genre", 
        y="Percentage", 
        color="Dataset",
        barmode="group",
        title="Genre Distribution Comparison",
        color_discrete_map={"Full Dataset": "#7194C2", "Sample Dataset": "#FF6B6B"}
    )
    
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Percentage of Movies (%)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.info(f"""
    **Target genres for sampling: {', '.join(viz_data['target_genres'])}** ⭐
    
    **Understanding the chart:**
    - Shows the percentage of movies that contain each genre
    - Sample dataset contains movies that have ANY of the target genres (Family, Mystery, Western)
    - Movies can belong to multiple genres, so percentages don't sum to 100%
    - Target genres are marked with ⭐
    """)
    
    # Show sampling statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Full Dataset", f"{full_total:,} movies")
    with col2:
        st.metric("Sample Dataset", f"{sample_total:,} movies")
    with col3:
        sample_ratio = (sample_total / full_total) * 100 if full_total > 0 else 0
        st.metric("Sample Ratio", f"{sample_ratio:.1f}%")

def display_dimension_reduction_explanation() -> None:
    """Display an explanation of the dimension reduction techniques."""
    with st.expander("About Dimension Reduction Techniques", expanded=False):
        st.write("""
        ### Understanding Dimension Reduction Techniques
        
        The visualizations use mathematical techniques to project high-dimensional movie data into 2D/3D space:
        
        #### t-SNE (t-Distributed Stochastic Neighbor Embedding)
        - Preserves local similarities between movies
        - Similar movies appear close together in clusters
        - Good at revealing the structure of the data
        - May take longer to compute for large datasets
        
        #### PCA (Principal Component Analysis)
        - Linear technique that maximizes variance along each dimension
        - Shows the primary directions of variation in the dataset
        - Faster than t-SNE but may miss non-linear relationships
        - Better at preserving global structure
        
        #### UMAP (Uniform Manifold Approximation and Projection)
        - Combines benefits of t-SNE and PCA
        - Preserves both local and global structure
        - Faster than t-SNE for large datasets
        - Good balance of performance and quality
        
        ### Reading the Visualization
        
        - **Grey points**: Full dataset movies
        - **Red points**: Sample dataset movies
        - **Clusters**: Movies that share similar characteristics
        - **Hover**: See details about each movie
        
        The distribution of red points shows how well the sampling method captures the diversity of the full dataset.
        """)
