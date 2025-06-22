"""
Utility functions for loading and preprocessing data for the RAG dashboard.
"""

import pandas as pd
import json
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import project configuration
from specialization.config.config import EVALUATION_INSIGHTS_PATH, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_insight_data():
    """
    Load evaluation insights data from the specified JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing the evaluation insights data.
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Construct the absolute path to the insights file
        insights_path = project_root / EVALUATION_INSIGHTS_PATH
        
        logging.info(f"Looking for insights file at: {insights_path}")
        
        if not insights_path.exists():
            logging.error(f"Evaluation insights file not found at {insights_path}")
            return None
            
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
            
        # Convert JSON data to DataFrame    
        insights_df = pd.DataFrame(insights_data)
        
        # Convert timestamp string to datetime if it exists
        if 'timestamp' in insights_df.columns:
            insights_df['timestamp'] = pd.to_datetime(insights_df['timestamp'])
            
        # Sort by timestamp
        if 'timestamp' in insights_df.columns:
            insights_df = insights_df.sort_values('timestamp', ascending=False)
            
        logging.info(f"Successfully loaded {len(insights_df)} evaluation records.")
        return insights_df
        
    except Exception as e:
        logging.error(f"Error loading insights data: {e}")
        return None
