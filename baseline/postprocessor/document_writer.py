"""
Module for file output and document saving functionality.
This file contains the DocumentWriter class that provides:
1. Utilities for saving processed data to various file formats
2. File path handling and directory creation capabilities
3. Standardized output formatting for consistent data storage
"""
import csv
import pandas as pd
from filelock import FileLock, Timeout


class DocumentWriter:
    """
    Utility class for writing documents and data to various file formats.
    Provides static methods for common file output operations.
    """
    
    @staticmethod
    def df_to_csv(df, directory, file_name):
        """
        Save the DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            directory (pathlib.Path or str): The directory to save the file in.
            file_name (str): The name of the file to save.
        """
        # Convert directory to Path object if it's a string
        from pathlib import Path
        if isinstance(directory, str):
            directory = Path(directory)
            
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        
        file_path = directory / f"{file_name}.csv"

        df.to_csv(
            file_path,
            index=False,                  # Do NOT write the index unless needed
            quoting=csv.QUOTE_ALL,        # Quote all fields (handles commas, quotes, newlines)
            quotechar='"',                # Standard quote character for CSV
            escapechar='\\',              # Escape character for quotechar if needed
            encoding='utf-8-sig',         # Ensures compatibility with Excel
            lineterminator='\n',         # Uniform newlines across platforms
        )
        
    @staticmethod
    def df_to_json(df, directory, file_name, append=True):
        """
        Save the DataFrame to a JSON file.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            directory (pathlib.Path or str): The directory to save the file in.
            file_name (str): The name of the file to save.
            append (bool): Whether to append to the file if it exists.
        """
        # Convert directory to Path object if it's a string
        from pathlib import Path
        if isinstance(directory, str):
            directory = Path(directory)
            
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        
        file_path = directory / f"{file_name}.json"
        
        # Create a file lock to prevent concurrent writes
        lock_path = file_path.with_suffix('.lock')
        
        try:
            with FileLock(str(lock_path), timeout=10):
                # Attempt to read existing data if appending
                if append:
                    try:
                        existing_df = pd.read_json(file_path, orient='records')
                        # Robustly align columns before concatenation
                        all_columns = sorted(set(existing_df.columns).union(set(df.columns)))
                        existing_df = existing_df.reindex(columns=all_columns, fill_value=pd.NA).astype('object')
                        df = df.reindex(columns=all_columns, fill_value=pd.NA)  # keep df dtypes
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                    except FileNotFoundError:
                        combined_df = df
                        print(f"File {file_path} not found. Creating a new file.")
                else:
                    combined_df = df
                    
                    
                # Append new data to existing data
                combined_df.to_json(
                    file_path,
                    orient='records',             # Each record is a separate line
                    lines=False,                   # Write each record on a new line
                    force_ascii=False,            # Allow non-ASCII characters
                    date_format='iso',            # ISO format for dates
                    indent=2,                     # Pretty print with indentation
                )
        except Timeout:
            print(f"Could not acquire lock for {file_path}. Another process may be writing to it.")
            
        
        # if append:
        #     try:
        #        existing_df = pd.read_json(file_path, orient='records')
        #        combined_df = pd.concat([existing_df, df], ignore_index=True)
        #     except FileNotFoundError:
        #         combined_df = df
        #         print(f"File {file_path} not found. Creating a new file.")



