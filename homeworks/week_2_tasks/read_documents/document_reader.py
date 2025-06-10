import os
import pandas as pd

def chunk_text(text, chunk_size=200):
    """
    Split text into chunks of specified size.
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Size of each chunk
        
    Returns:
        list: List of text chunks
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def read_csv(file_path):
    """
    Read a CSV file and return the data as a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file (absolute or relative)
        
    Returns:
        pandas.DataFrame: DataFrame containing the CSV data
    """
    try:
        # Convert relative path to absolute path if needed
        abs_path = os.path.abspath(file_path)
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(abs_path)
        
        # Return the entire DataFrame
        return df
    
    except FileNotFoundError as e:
        print(f"File not found: {file_path}. Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()


