#!/usr/bin/env python3
"""
Raw to Processed Data Pipeline for Specialization Track

This pipeline processes raw CSV movie data from data/raw directory:
1. Loads CSV files using extended document reader
2. Joins multiple CSV files based on common ID column
3. Filters data by specific genres (Family, Mystery, Western)
4. Outputs processed data as JSON to data/processed directory

Features:
- Extends baseline document reader with CSV support
- Configurable genre filtering
- Robust data processing with error handling
- LangChain integration where applicable
"""

import os
import sys
import pandas as pd
from typing import List, Dict
from pathlib import Path
import logging

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.postprocessor.document_writer import DocumentWriter
from specialization.config.config import (
    RAW_DOCUMENT_DIR_PATH, PROCESSED_DOCUMENT_DIR_PATH, LOG_LEVEL, 
    TARGET_GENRES, PROCESSED_DOCUMENT_NAME, EXCLUDED_COLUMNS, FLATTEN_COLUMNS, RAW_DATA_FILES
)
from specialization.preprocessor.document_reader import SpecializedDocumentLoaderFactory
from specialization.utils import (
    exclude_columns, flatten_list_columns, extract_names_from_json_list,
    safe_numeric_conversion
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawToProcessedPipeline:
    """
    Pipeline for processing raw CSV movie data into filtered JSON output.
    """
    
    def __init__(self, raw_data_path: str = None, processed_data_path: str = None, 
                 target_genres: List[str] = None):
        """
        Initialize the pipeline with configurable paths and genres.
        
        Args:
            raw_data_path (str): Path to raw data directory
            processed_data_path (str): Path to processed data output directory  
            target_genres (List[str]): List of genres to filter for
        """
        self.raw_data_path = raw_data_path or RAW_DOCUMENT_DIR_PATH
        self.raw_fiels = RAW_DATA_FILES
        self.processed_data_path = processed_data_path or PROCESSED_DOCUMENT_DIR_PATH
        self.target_genres = target_genres or TARGET_GENRES
        self.document_loader = SpecializedDocumentLoaderFactory()
        
        # Ensure output directory exists
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized - Raw: {self.raw_data_path}, Processed: {self.processed_data_path}")
        logger.info(f"Target genres: {self.target_genres}")
    
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the raw data directory.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame
        """
        csv_files = {}
        raw_path = Path(self.raw_data_path)
        
        for file in self.raw_fiels:
            file_path = raw_path / file
            try:
                loader = self.document_loader.get_loader(str(file_path))
                df = loader.load(str(file_path))
                filename = file_path.stem  # Get filename without extension
                
                # Early exclusion of unwanted columns for performance
                df = exclude_columns(df, EXCLUDED_COLUMNS)
                
                csv_files[filename] = df
                logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
                
        return csv_files
    
    def parse_genres(self, genres_str: str) -> List[str]:
        """
        Parse genres from JSON-like string format to list of genre names.
        
        Args:
            genres_str (str): String representation of genres list
            
        Returns:
            List[str]: List of genre names
        """
        return extract_names_from_json_list(genres_str)
    
    def filter_by_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame by target genres.
        
        Args:
            df (pd.DataFrame): DataFrame with genres column
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if 'genres' not in df.columns:
            logger.warning("No 'genres' column found in DataFrame")
            return df
            
        # Parse genres and check for target genres
        def has_target_genre(genres_str):
            genres = self.parse_genres(genres_str)
            return any(genre in self.target_genres for genre in genres)
        
        mask = df['genres'].apply(has_target_genre)
        filtered_df = df[mask].copy()
        
        logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} rows for genres: {self.target_genres}")
        return filtered_df
    
    def join_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Join multiple DataFrames on common 'id' column.
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Dictionary of DataFrames to join
            
        Returns:
            pd.DataFrame: Joined DataFrame
        """
        if not dataframes:
            raise ValueError("No DataFrames provided for joining")
            
        # Start with the first DataFrame
        result_df = None
        
        for name, df in dataframes.items():
            if 'id' not in df.columns:
                logger.warning(f"DataFrame {name} missing 'id' column, skipping join")
                continue
                
            # Ensure 'id' column is consistent type
            df = safe_numeric_conversion(df, 'id')
            
            if result_df is None:
                result_df = df.copy()
                logger.info(f"Starting join with {name}: {len(result_df)} rows")
            else:
                # Perform left join to preserve movies_metadata as primary
                before_count = len(result_df)
                result_df = result_df.merge(df, on='id', how='left', suffixes=('', f'_{name}'))
                logger.info(f"Joined with {name}: {before_count} -> {len(result_df)} rows")
        
        if result_df is None:
            raise ValueError("No valid DataFrames found for joining")
            
        return result_df
    
    def process_data(self) -> pd.DataFrame:
        """
        Main processing method that orchestrates the entire pipeline.
        
        Returns:
            pd.DataFrame: Processed and filtered DataFrame
        """
        logger.info("Starting data processing pipeline")
        
        # Step 1: Load CSV files (with early column exclusion)
        dataframes = self.load_csv_files()
        if not dataframes:
            raise ValueError("No CSV files found in raw data directory")
        
        # Step 2: Filter movies_metadata by genres first (before joining for efficiency)
        if 'movies_metadata' in dataframes:
            dataframes['movies_metadata'] = self.filter_by_genres(dataframes['movies_metadata'])
        else:
            logger.warning("movies_metadata.csv not found - cannot filter by genres")
        
        # Step 3: Join all DataFrames
        joined_df = self.join_dataframes(dataframes)
        
        # Step 4: Flatten specified columns
        joined_df = flatten_list_columns(joined_df, FLATTEN_COLUMNS)
        
        logger.info(f"Final processed dataset: {len(joined_df)} rows, {len(joined_df.columns)} columns")
        return joined_df
    
    def save_to_json(self, df: pd.DataFrame, filename: str = "processed_movies_data.json") -> str:
        """
        Save processed DataFrame to JSON format.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        filename = filename or PROCESSED_DOCUMENT_NAME            
        output_path = Path(self.processed_data_path) / filename
        
        try:
            # Use DocumentWriter for consistent output formatting
            DocumentWriter.df_to_json(df, self.processed_data_path, filename.replace('.json', ''), append=False)
            logger.info(f"Successfully saved processed data to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save data to {output_path}: {e}")
            raise
    
    def run(self) -> str:
        """
        Execute the complete pipeline.
        
        Returns:
            str: Path to the output JSON file
        """
        try:
            # Process the data
            processed_df = self.process_data()
            
            # Save to JSON
            output_path = self.save_to_json(processed_df)
            
            logger.info("Pipeline completed successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run the pipeline."""
    try:
        pipeline = RawToProcessedPipeline()
        output_path = pipeline.run()
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()