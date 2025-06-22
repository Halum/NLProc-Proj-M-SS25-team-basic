#!/usr/bin/env python3
"""
Main Orchestrator for Specialization Track Pipelines

This script serves as the main entry point for running the specialization pipelines.
It provides a command-line interface to:
1. Run the raw_to_processed pipeline to convert CSV data to processed JSON
2. Run the processed_to_embeddings pipeline to create vector embeddings
3. Run the evaluation_pipeline to measure system performance
4. Run the user_query pipeline for interactive querying
5. Run all pipelines in sequence (except user_query)

Features:
- Dynamic module loading for hot reloading of pipeline code
- Interactive command-line interface with clear options
- Sequential pipeline execution with proper error handling
- Graceful handling of nested loops for user_query pipeline
"""

import os
import sys
import importlib
import logging

# Get paths for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # src directory
project_dir = os.path.dirname(src_dir)  # project root directory

# Make sure the paths are in sys.path with the correct order
# Project root should be first, allowing proper package resolution
paths = [project_dir, src_dir]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.debug(f"Python path: {sys.path}")



def import_pipeline_module(module_name):
    """
    Helper function to import pipeline modules with fallback.
    First tries as a standard import, then tries various approaches.
    """
    try:
        # First try with the standard import path
        module = importlib.import_module(f'src.specialization.pipelines.{module_name}')
        return module
    except ImportError:
        try:
            # Second approach: try relative to src
            module = importlib.import_module(f'specialization.pipelines.{module_name}')
            return module
        except ImportError:
            # Third approach: try with direct filename
            module = importlib.import_module(f'{module_name}', package='specialization.pipelines')
            return module

def clear_screen():
    """Clear the terminal screen based on OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    clear_screen()
    print("=" * 80)
    print("                  NLP PROJECT - SPECIALIZATION TRACK PIPELINES")
    print("=" * 80)
    print()

def print_menu():
    """Print the main menu options."""
    print("\nSelect a pipeline to execute:")
    print("1. Raw to Processed - Convert raw CSV files to processed JSON")
    print("2. Processed to Embeddings - Create vector embeddings from processed data")
    print("3. Evaluation Pipeline - Run system evaluation with gold standard data")
    print("4. User Query Pipeline - Interactive query mode")
    print("5. Run All Pipelines (1-3 in sequence)")
    print("q. Quit")
    print("\nEnter your choice: ", end="")

def run_raw_to_processed_pipeline():
    """Run the raw to processed pipeline with dynamic import."""
    try:
        # Dynamically import the module using our helper function
        raw_to_processed = import_pipeline_module('raw_to_processed')
        # Reload if it was already imported
        importlib.reload(raw_to_processed)
        
        logger.info("Starting Raw to Processed Pipeline...")
        raw_to_processed.main()
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error in Raw to Processed Pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("\nPress Enter to continue...")

def run_processed_to_embeddings_pipeline():
    """Run the processed to embeddings pipeline with dynamic import."""
    try:
        # Dynamically import the module using our helper function
        processed_to_embeddings = import_pipeline_module('processed_to_embeddings')
        # Reload if it was already imported
        importlib.reload(processed_to_embeddings)
        
        logger.info("Starting Processed to Embeddings Pipeline...")
        processed_to_embeddings.main()
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error in Processed to Embeddings Pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("\nPress Enter to continue...")

def run_evaluation_pipeline():
    """Run the evaluation pipeline with dynamic import."""
    try:
        # Dynamically import the module using our helper function
        evaluation_pipeline = import_pipeline_module('evaluation_pipeline')
        # Reload if it was already imported
        importlib.reload(evaluation_pipeline)
        
        logger.info("Starting Evaluation Pipeline...")
        evaluation_pipeline.main()
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error in Evaluation Pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("\nPress Enter to continue...")

def run_user_query_pipeline():
    """
    Run the user query pipeline with dynamic import.
    This function handles the nested loop within user_query.
    """
    try:
        # Dynamically import the module using our helper function
        user_query = import_pipeline_module('user_query')
        # Reload if it was already imported
        importlib.reload(user_query)
        
        logger.info("Starting User Query Pipeline...")
        print("\nEntering interactive query mode. Type 'exit' to return to main menu.")
        user_query.main()
        # The user_query pipeline has its own loop that exits when user types 'exit'
    except Exception as e:
        logger.error(f"Error in User Query Pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("\nPress Enter to continue...")

def run_all_pipelines():
    """Run all pipelines in sequence (except user_query)."""
    try:
        logger.info("Running all pipelines in sequence...")
        
        # Run pipelines in order
        run_raw_to_processed_pipeline()
        run_processed_to_embeddings_pipeline()
        run_evaluation_pipeline()
        
        logger.info("All pipelines completed!")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error running all pipelines: {e}")
        input("\nPress Enter to continue...")

def main():
    """Main function to orchestrate the pipelines."""
    while True:
        print_header()
        print_menu()
        choice = input().strip().lower()
        
        if choice == '1':
            run_raw_to_processed_pipeline()
        elif choice == '2':
            run_processed_to_embeddings_pipeline()
        elif choice == '3':
            run_evaluation_pipeline()
        elif choice == '4':
            run_user_query_pipeline()
        elif choice == '5':
            run_all_pipelines()
        elif choice == 'q':
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
