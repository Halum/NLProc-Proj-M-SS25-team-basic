"""
Module for file output and document saving functionality.
This file contains the DocumentWriter class that provides:
1. Utilities for saving processed data to various file formats
2. File path handling and directory creation capabilities
3. Standardized output formatting for consistent data storage
"""

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
        df.to_csv(file_path, index=False)