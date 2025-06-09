#!/usr/bin/env python3
"""
Data processing utilities for the specialization track.

This module contains reusable utility functions for data manipulation,
cleaning, and transformation operations that can be used across different
pipelines and components.
"""

import pandas as pd
import ast
import logging
from typing import List

logger = logging.getLogger(__name__)


def exclude_columns(df: pd.DataFrame, columns_to_exclude: List[str]) -> pd.DataFrame:
    """
    Exclude specified columns from DataFrame for performance optimization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_exclude (List[str]): List of column names to exclude
        
    Returns:
        pd.DataFrame: DataFrame with specified columns removed
    """
    if not columns_to_exclude:
        return df
        
    existing_columns = [col for col in columns_to_exclude if col in df.columns]
    
    if existing_columns:
        df = df.drop(columns=existing_columns)
        logger.info(f"Excluded columns: {existing_columns}")
    
    return df


def flatten_list_columns(df: pd.DataFrame, columns_to_flatten: List[str]) -> pd.DataFrame:
    """
    Flatten specified columns that contain JSON-like list data to extract only names.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_flatten (List[str]): List of column names to flatten
        
    Returns:
        pd.DataFrame: DataFrame with flattened columns
    """
    if not columns_to_flatten:
        return df
        
    df = df.copy()
    
    for column in columns_to_flatten:
        if column in df.columns:
            df[column] = df[column].apply(extract_names_from_json_list)
            logger.info(f"Flattened column: {column}")
    
    return df


def extract_names_from_json_list(json_str: str) -> List[str]:
    """
    Extract names from JSON-like string containing list of dictionaries.
    
    This function handles various name fields like 'name', 'english_name', 'title'.
    
    Args:
        json_str (str): JSON string representation of list
        
    Returns:
        List[str]: List of extracted names
    """
    if pd.isna(json_str) or not json_str:
        return []
        
    try:
        # Safely evaluate the string as a Python literal
        items_list = ast.literal_eval(json_str)
        if isinstance(items_list, list):
            names = []
            for item in items_list:
                if isinstance(item, dict):
                    # Handle different possible name fields
                    name = item.get('name') or item.get('english_name') or item.get('title', '')
                    if name:
                        names.append(name)
            return names
        return []
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse JSON list: {json_str[:50]}... Error: {e}")
        return []


def parse_json_field(json_str: str, field_name: str = 'name') -> List[str]:
    """
    Parse a JSON-like string field and extract specific field values.
    
    Args:
        json_str (str): String representation of JSON list
        field_name (str): Field name to extract from each dictionary
        
    Returns:
        List[str]: List of extracted field values
    """
    if pd.isna(json_str) or not json_str:
        return []
        
    try:
        # Safely evaluate the string as a Python literal
        items_list = ast.literal_eval(json_str)
        if isinstance(items_list, list):
            return [item.get(field_name, '') for item in items_list 
                   if isinstance(item, dict) and item.get(field_name)]
        return []
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse JSON field {field_name}: {json_str[:50]}... Error: {e}")
        return []


def safe_numeric_conversion(df: pd.DataFrame, column: str, errors: str = 'coerce') -> pd.DataFrame:
    """
    Safely convert a DataFrame column to numeric type.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to convert
        errors (str): How to handle conversion errors ('coerce', 'raise', 'ignore')
        
    Returns:
        pd.DataFrame: DataFrame with converted column
    """
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_numeric(df[column], errors=errors)
        if errors == 'coerce':
            # Remove rows with invalid conversions (NaN values)
            df = df.dropna(subset=[column])
        logger.info(f"Converted column '{column}' to numeric type")
    else:
        logger.warning(f"Column '{column}' not found in DataFrame")
    
    return df


def filter_dataframe_by_list_field(df: pd.DataFrame, column: str, target_values: List[str], 
                                 field_name: str = 'name') -> pd.DataFrame:
    """
    Filter DataFrame based on whether any target values exist in a JSON list field.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column containing JSON list data
        target_values (List[str]): Values to filter for
        field_name (str): Field name to check within JSON objects
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df
        
    def has_target_value(json_str):
        parsed_values = parse_json_field(json_str, field_name)
        return any(value in target_values for value in parsed_values)
    
    mask = df[column].apply(has_target_value)
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtered {len(df)} rows to {len(filtered_df)} rows for {field_name} values: {target_values}")
    return filtered_df
