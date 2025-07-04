"""
Utility functions for loading and processing historical data from insight files.

This module now uses the centralized data transformation utilities from data_transformation.py
"""

import logging
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import project configuration
from specialization.config.config import LOG_LEVEL

# Import data transformation utilities
from specialization.streamlit.utils.data_transformation import prepare_historical_metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_historical_metrics():
    """
    Process all insight files and extract average metrics for trending.
    
    Returns:
        pd.DataFrame: DataFrame with historical metrics
    """
    # Use the data transformation function
    result = prepare_historical_metrics()
    
    return result
