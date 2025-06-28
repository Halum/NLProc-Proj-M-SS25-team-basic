#!/usr/bin/env python3
"""
Script to run the Streamlit RAG Performance Dashboard.
This script sets up the path and environment correctly before launching Streamlit.
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Get the absolute path to this script
    script_path = Path(__file__).resolve()
    
    # Get streamlit directory (parent of this script)
    streamlit_dir = script_path.parent
    
    # Get project root (3 levels up from this script)
    project_root = streamlit_dir.parent.parent.parent
    
    logging.info(f"Streamlit directory: {streamlit_dir}")
    logging.info(f"Project root: {project_root}")
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    
    # Check if the insights file exists
    insights_path = project_root / "specialization" / "data" / "insight" / "evaluation_insights.json"
    logging.info(f"Checking for insights file at: {insights_path}")
    
    if not insights_path.exists():
        logging.warning(f"Insights file not found at {insights_path}")
        logging.info("You may need to run the evaluation pipeline first to generate insights.")
    else:
        logging.info(f"Found insights file at {insights_path}")
    
    # Change to the streamlit directory
    os.chdir(streamlit_dir)
    
    # Run Streamlit
    logging.info("Starting Streamlit app...")
    streamlit_cmd = ["streamlit", "run", "app.py"]
    subprocess.run(streamlit_cmd)

if __name__ == "__main__":
    main()
