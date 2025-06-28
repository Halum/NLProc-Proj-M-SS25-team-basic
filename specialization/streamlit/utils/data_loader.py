"""
Utility functions for loading and preprocessing data for the RAG dashboard.
"""

import pandas as pd
import json
import logging
import sys
import os
import re
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import project configuration
from specialization.config.config import EVALUATION_INSIGHTS_PATH, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_available_insight_files():
    """
    Gets all available insight files in the insight directory.
    Only returns files that have timestamps in their names.
    
    Returns:
        list: List of tuples containing (file path, display name, timestamp) sorted by timestamp descending
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Get the insight directory
        insight_dir = os.path.dirname(project_root / EVALUATION_INSIGHTS_PATH)
        insight_base_name = os.path.splitext(os.path.basename(EVALUATION_INSIGHTS_PATH))[0]
        
        # Get all JSON files in the directory that match the pattern - only with timestamps
        pattern = re.compile(f"{insight_base_name}_(\\d{{8}}_\\d{{6}})\\.json$")
        files = []
        
        if os.path.exists(insight_dir):
            for filename in os.listdir(insight_dir):
                if filename.endswith('.json'):
                    match = pattern.match(filename)
                    if match:
                        timestamp_str = match.group(1)
                        
                        # Parse the timestamp from filename
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        display_name = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            
                        file_path = os.path.join(insight_dir, filename)
                        files.append((file_path, display_name, timestamp))
        
        # Sort files by timestamp (newest first)
        files.sort(key=lambda x: x[2], reverse=True)
        
        logging.info(f"Found {len(files)} insight files with timestamps")
        return files
    
    except Exception as e:
        logging.error(f"Error getting available insight files: {e}")
        return []

def load_insight_data(file_path=None):
    """
    Load evaluation insights data from the specified JSON file.
    
    Args:
        file_path (str, optional): Path to the specific insight file to load.
                                  If None, loads the most recent file.
    
    Returns:
        pd.DataFrame: DataFrame containing the evaluation insights data.
    """
    try:
        # If no specific file path is provided, use the most recent one
        if file_path is None:
            available_files = get_available_insight_files()
            
            if not available_files:
                # No timestamped files found
                logging.error("No timestamped insight files available")
                return None
            else:
                # Use the most recent file
                file_path = available_files[0][0]
        
        logging.info(f"Loading insights file: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"Evaluation insights file not found at {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
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
