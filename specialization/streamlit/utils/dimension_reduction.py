"""
Dimension reduction utilities for visualizing high-dimensional movie data in 2D/3D space.

This module contains functions for preparing and transforming movie data
using dimensionality reduction techniques like t-SNE, PCA, and UMAP.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import os

# Machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import UMAP conditionally to handle potential installation issues
try:
    import umap.umap_ as umap
except ImportError:
    # UMAP is optional, will show error message to user when selected
    umap = None

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import project configuration
try:
    from specialization.config.config import (
        PROCESSED_DOCUMENT_DIR_PATH, 
        LOG_LEVEL,
        TARGET_GENRES,
        DATA_SAMPLE_SIZE,
        CACHE_DIR_PATH
    )
except ImportError:
    # Fallback if CACHE_DIR_PATH is not available
    from specialization.config.config import (
        PROCESSED_DOCUMENT_DIR_PATH, 
        LOG_LEVEL,
        TARGET_GENRES,
        DATA_SAMPLE_SIZE
    )
    CACHE_DIR_PATH = "specialization/data/processed/cache/"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_movie_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both the full movie dataset and sample dataset.
    
    Returns:
        Tuple containing (full_dataset, sample_dataset) as pandas DataFrames
    """
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Full dataset path
    full_data_path = project_root / PROCESSED_DOCUMENT_DIR_PATH / "processed_movies_data.json"
    
    # Sample dataset path
    sample_data_path = project_root / PROCESSED_DOCUMENT_DIR_PATH / "processed_movies_data_sample.json"
    
    try:
        # Load full dataset
        with open(full_data_path, 'r') as f:
            full_data = json.load(f)
        full_df = pd.DataFrame(full_data)
        
        # Load sample dataset
        with open(sample_data_path, 'r') as f:
            sample_data = json.load(f)
        sample_df = pd.DataFrame(sample_data)
        
        logger.info(f"Loaded full dataset: {len(full_df)} movies")
        logger.info(f"Loaded sample dataset: {len(sample_df)} movies")
        
        return full_df, sample_df
    
    except Exception as e:
        logger.error(f"Error loading movie data: {e}")
        raise

def preprocess_movie_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess movie features for dimensionality reduction.
    
    Args:
        df: DataFrame containing movie data
        
    Returns:
        DataFrame with preprocessed features ready for dimension reduction
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle numeric columns - ensure we handle strings that can't be converted
    numeric_columns = ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count']
    for col in numeric_columns:
        if col in processed_df.columns:
            # First try to clean up the data - handle non-numeric values
            processed_df[col] = processed_df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
            # Convert to numeric, coerce errors to NaN
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Extract year from release_date
    if 'release_date' in processed_df.columns:
        # Convert to string first to handle potential non-string values
        processed_df['release_date'] = processed_df['release_date'].astype(str)
        # Filter out obviously non-date strings
        date_pattern = r'^\d{4}[-/]?\d{1,2}[-/]?\d{1,2}$'
        processed_df.loc[~processed_df['release_date'].str.contains(date_pattern, na=False), 'release_date'] = None
        
        processed_df['release_year'] = pd.to_datetime(
            processed_df['release_date'], 
            errors='coerce'
        ).dt.year
    
    # Handle genres (convert list to one-hot encoding) - with better error handling
    if 'genres' in processed_df.columns:
        # Some rows might have genres as strings rather than lists
        def safe_convert_genre(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                if x.startswith('['):
                    try:
                        return eval(x)
                    except Exception:
                        return []
            return []
        
        processed_df['genres'] = processed_df['genres'].apply(safe_convert_genre)
        
        # Flatten genres and create one-hot encoding
        all_genres = set()
        for genres in processed_df['genres']:
            if isinstance(genres, list):
                all_genres.update(genres)
        
        for genre in all_genres:
            processed_df[f'genre_{genre}'] = processed_df['genres'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
    
    # Runtime as numeric
    if 'runtime' in processed_df.columns:
        # Clean up non-numeric characters first
        processed_df['runtime'] = processed_df['runtime'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        processed_df['runtime'] = pd.to_numeric(processed_df['runtime'], errors='coerce')
    
    # Drop columns with too many missing values or text columns not used for clustering
    columns_to_drop = [
        'overview', 'title', 'original_title', 'adult', 'status', 'tagline',
        'genres', 'release_date', 'cast', 'keywords',
        'production_companies', 'production_countries', 'spoken_languages'
    ]
    
    processed_df = processed_df.drop([col for col in columns_to_drop if col in processed_df.columns], axis=1)
    
    # Fill missing values
    for col in processed_df.columns:
        if pd.api.types.is_numeric_dtype(processed_df[col]):
            # Check if column is entirely NaN
            if processed_df[col].isna().all():
                processed_df[col] = 0  # Default to zero if all are NaN
            else:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    logger.info(f"Preprocessed movie features, resulting in {processed_df.shape[1]} features")
    return processed_df

def prepare_for_dimension_reduction(full_df: pd.DataFrame, sample_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare both datasets for dimension reduction by standardizing features.
    
    Args:
        full_df: Full movie dataset
        sample_df: Sample movie dataset
        
    Returns:
        Tuple containing (standardized_combined_data, full_indices, sample_indices)
    """
    # Extract features for dimensionality reduction
    full_features = preprocess_movie_features(full_df)
    sample_features = preprocess_movie_features(sample_df)
    
    # Make sure both dataframes have the same columns
    common_columns = list(set(full_features.columns) & set(sample_features.columns))
    full_features = full_features[common_columns]
    sample_features = sample_features[common_columns]
    
    # Get indices for tracking
    full_indices = np.arange(len(full_features))
    sample_indices = np.arange(len(full_features), len(full_features) + len(sample_features))
    
    # Combine datasets for joint scaling
    combined_features = pd.concat([full_features, sample_features])
    
    # Standardize features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(combined_features)
    
    logger.info(f"Prepared {len(standardized_data)} movies for dimension reduction with {len(common_columns)} features")
    return standardized_data, full_indices, sample_indices

def reduce_dimensions(data: np.ndarray, method: str = 'tsne', n_components: int = 2, 
                     random_state: int = 42) -> np.ndarray:
    """
    Reduce dimensionality of the data using the specified method.
    
    Args:
        data: Input data matrix
        method: Dimension reduction method ('tsne', 'pca', or 'umap')
        n_components: Number of components (2 for 2D, 3 for 3D)
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced data array
    """
    if method.lower() == 'tsne':
        reducer = TSNE(
            n_components=n_components, 
            perplexity=min(30, len(data) - 1),  # Adjust perplexity for small datasets
            random_state=random_state,
            init='pca'
        )
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'umap':
        if umap is None:
            raise ImportError(
                "UMAP is not installed. Please install it with 'pip install umap-learn'"
            )
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            min_dist=0.1,
            n_neighbors=min(15, len(data) - 1)  # Adjust neighbors for small datasets
        )
    else:
        raise ValueError(f"Unknown dimension reduction method: {method}")
    
    reduced_data = reducer.fit_transform(data)
    logger.info(f"Reduced dimensions to {n_components}D using {method}")
    return reduced_data

def prepare_cluster_visualization_data(n_components: int = 2, method: str = 'tsne',
                                      random_state: int = 42, cache: bool = True) -> Dict:
    """
    Prepare data for cluster visualization, with caching to avoid recomputing.
    
    Args:
        n_components: Number of dimensions (2 or 3)
        method: Dimension reduction method ('tsne', 'pca', or 'umap')
        random_state: Random seed for reproducibility
        cache: Whether to cache results for future use
    
    Returns:
        Dictionary with visualization data including points and metadata
    """
    # Define cache path using config
    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / CACHE_DIR_PATH
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = cache_dir / f"vis_data_{method}_{n_components}d.json"
    
    # Try to load from cache
    if cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                logger.info(f"Loaded visualization data from cache: {cache_file}")
                return cache_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, recomputing...")
    
    try:
        # Load data
        full_df, sample_df = load_movie_data()
        
        # Fall back to PCA if UMAP is requested but not available
        if method.lower() == 'umap' and umap is None:
            logger.warning("UMAP is not installed. Falling back to PCA.")
            method = 'pca'
            
        # Prepare data for dimension reduction
        standardized_data, full_indices, sample_indices = prepare_for_dimension_reduction(full_df, sample_df)
        
        # Reduce dimensions
        reduced_data = reduce_dimensions(
            standardized_data, 
            method=method,
            n_components=n_components,
            random_state=random_state
        )
        
        # Prepare visualization data
        full_points = reduced_data[full_indices].tolist()
        sample_points = reduced_data[sample_indices].tolist()
        
        # Prepare metadata for tooltips with better error handling
        full_metadata = []
        for i, idx in enumerate(full_df.index):
            try:
                meta = {
                    "id": int(float(str(full_df.loc[idx, "id"]).replace(',', ''))) if "id" in full_df.columns else i,
                    "title": str(full_df.loc[idx, "original_title"]) if "original_title" in full_df.columns else "Unknown",
                    "genres": full_df.loc[idx, "genres"] if "genres" in full_df.columns else [],
                    "year": None,
                    "is_sample": False
                }
                
                # Safe extraction of year
                if "release_date" in full_df.columns:
                    try:
                        year = pd.to_datetime(full_df.loc[idx, "release_date"], errors="coerce").year
                        meta["year"] = None if pd.isna(year) else int(year)
                    except Exception:
                        meta["year"] = None
                        
                full_metadata.append(meta)
            except Exception as e:
                logger.warning(f"Error preparing metadata for movie {idx}: {e}")
                # Add minimal metadata to keep alignment
                full_metadata.append({
                    "id": i,
                    "title": "Unknown",
                    "genres": [],
                    "year": None,
                    "is_sample": False
                })
        
        # Prepare sample metadata
        sample_metadata = []
        for i, idx in enumerate(sample_df.index):
            try:
                meta = {
                    "id": int(float(str(sample_df.loc[idx, "id"]).replace(',', ''))) if "id" in sample_df.columns else i,
                    "title": str(sample_df.loc[idx, "original_title"]) if "original_title" in sample_df.columns else "Unknown",
                    "genres": sample_df.loc[idx, "genres"] if "genres" in sample_df.columns else [],
                    "year": None,
                    "is_sample": True
                }
                
                # Safe extraction of year
                if "release_date" in sample_df.columns:
                    try:
                        year = pd.to_datetime(sample_df.loc[idx, "release_date"], errors="coerce").year
                        meta["year"] = None if pd.isna(year) else int(year)
                    except Exception:
                        meta["year"] = None
                        
                sample_metadata.append(meta)
            except Exception as e:
                logger.warning(f"Error preparing metadata for sample movie {idx}: {e}")
                # Add minimal metadata to keep alignment
                sample_metadata.append({
                    "id": i,
                    "title": "Unknown",
                    "genres": [],
                    "year": None,
                    "is_sample": True
                })
        
        # Create result
        result = {
            "method": method,
            "dimensions": n_components,
            "full_points": full_points,
            "sample_points": sample_points,
            "full_metadata": full_metadata,
            "sample_metadata": sample_metadata,
            "target_genres": TARGET_GENRES,
            "sample_size": DATA_SAMPLE_SIZE,
            "full_size": len(full_df),
            "actual_sample_size": len(sample_df)
        }
        
        # Cache results
        if cache:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                logger.info(f"Cached visualization data to: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to cache results: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error preparing visualization data: {str(e)}")
        # Return a minimal valid structure for graceful failure
        return {
            "method": method,
            "dimensions": n_components,
            "full_points": [],
            "sample_points": [],
            "full_metadata": [],
            "sample_metadata": [],
            "target_genres": TARGET_GENRES,
            "sample_size": DATA_SAMPLE_SIZE,
            "full_size": 0,
            "actual_sample_size": 0,
            "error": str(e)
        }
