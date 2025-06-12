#!/usr/bin/env python3
"""
Data processing utilities for the specialization track.

This module contains reusable utility functions for data manipulation,
cleaning, and transformation operations that can be used across different
pipelines and components.
"""

from math import e
import pandas as pd
import ast
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def filter_json_columns(data: List[Dict[str, Any]], columns_to_keep: List[str], columns_type_mapping: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Filter JSON data to keep only specified columns.
    
    This function is the complement to exclude_columns but works with JSON data
    instead of pandas DataFrames. It filters a list of dictionaries to keep only
    the specified columns.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries (JSON data)
        columns_to_keep (List[str]): List of column names to keep
        columns_type_mapping (List[Dict[str, str]]): Mapping of column names to data types
        
    Returns:
        List[Dict[str, Any]]: Filtered data with only specified columns
    """
    if not columns_to_keep:
        return data
    
    # Prepare a quick lookup for type mapping
    type_map = {entry['column']: entry['type'] for entry in columns_type_mapping}
    
    # Type casting functions
    def cast_value(value, expected_type):
        try:
            if expected_type == 'int':
                return int(float(value)) if value not in [None, ''] else None
            elif expected_type == 'float':
                return float(value) if value not in [None, ''] else None
            elif expected_type == 'str':
                return str(value) if value is not None else None
            elif expected_type == 'bool':
                return bool(value) if isinstance(value, (int, bool)) else str(value).lower() in ['true', '1']
            elif expected_type == 'year':
                return datetime.strptime(value, '%Y-%m-%d').year if isinstance(value, str) and value else None
            else:
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast value '{value}' to {expected_type}: {e}")
            return None
        
    filtered_data = []
    for item in data:
        filtered_item = {}
        
        # for column in columns_to_keep:
        #     if column in item:
        #         print(f"Column '{column}' found and '{item[column]}'.")
        #         filtered_item[column] = item[column]
        #     else:
        #         logger.warning(f"Column '{column}' not found in item with id: {item.get('id', 'unknown')}")
        #         filtered_item[column] = None
        for column in columns_to_keep:
            if column in item:
                value = item[column]
                
                # Cast value to expected type if specified
                if column in type_map:
                    expected_type = type_map[column]
                    value = cast_value(value, expected_type)
                    
                filtered_item[column] = value
            else:
                logger.warning(f"Column '{column}' not found in item with id: {item.get('id', 'unknown')}")
                filtered_item[column] = None
        
        # Only include items that have the required column with a value
        filtered_data.append(filtered_item)
    
    logger.info(f"Filtered {len(data)} items to {len(filtered_data)} items with columns: {columns_to_keep}")
    return filtered_data


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


def sample_dataframe(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly sample rows from a DataFrame for MVP/development purposes.
    
    Args:
        df (pd.DataFrame): Input DataFrame to sample from
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducible results
        
    Returns:
        pd.DataFrame: Sampled DataFrame with reset index
    """
    if sample_size <= 0:
        logger.warning("Sample size must be positive, returning original DataFrame")
        return df
        
    if len(df) <= sample_size:
        logger.info(f"DataFrame has {len(df)} rows, which is <= sample_size ({sample_size}). Returning original DataFrame.")
        return df
        
    logger.info(f"Sampling {sample_size} rows from {len(df)} rows for development")
    sampled_df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    logger.info(f"Sampled dataset size: {len(sampled_df)} rows")
    
    return sampled_df


def concatenate_columns_to_chunking(data: List[Dict[str, Any]], chunking_column: str, 
                                  columns_to_add: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Concatenate additional columns to the chunking column using specified prefixes.
    
    This function enhances the main chunking column by appending content from other
    columns with customizable prefixes. This is useful for enriching the text that
    will be used for chunking and embedding generation.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries (JSON data)
        chunking_column (str): Name of the main column used for chunking
        columns_to_add (List[Dict[str, str]]): List of column configurations with format:
            [{'column': 'column_name', 'prefix': 'Prefix text '}]
            
    Returns:
        List[Dict[str, Any]]: Data with enhanced chunking column
        
    Example:
        >>> data = [{'overview': 'A great movie', 'cast': 'Actor1, Actor2'}]
        >>> columns_to_add = [{'column': 'cast', 'prefix': 'Starring: '}]
        >>> result = concatenate_columns_to_chunking(data, 'overview', columns_to_add)
        >>> result[0]['overview']  # 'A great movie. Starring: Actor1, Actor2'
    """
    if not columns_to_add or not chunking_column:
        logger.info("No columns to concatenate or chunking column not specified")
        return data
    
    enhanced_data = []
    successful_concatenations = 0
    chunking_column_no_data = 0
    
    for item in data:
        enhanced_item = item.copy()
        
        # Get the base chunking column content
        base_content = item.get(chunking_column, '')
        if not base_content:
            logger.warning(f"Item missing chunking column '{chunking_column}': {item.get('title', 'unknown')}")
            chunking_column_no_data += 1
            # enhanced_data.append(enhanced_item)
            # continue
            
        # Build the enhanced content
        enhanced_content = str(base_content).strip()
        
        for column_config in columns_to_add:
            column_name = column_config.get('column')
            prefix = column_config.get('prefix', '')
            
            if not column_name:
                logger.warning(f"Column name not specified in config: {column_config}")
                continue
                
            # Get the additional column content
            additional_content = item.get(column_name)
            if additional_content:
                # Handle list content (like cast arrays)
                if isinstance(additional_content, list):
                    additional_text = ', '.join(str(item) for item in additional_content if item)
                else:
                    additional_text = str(additional_content).strip()
                
                if additional_text:
                    # Add separator and prefix with additional content
                    separator = '. ' if enhanced_content and not enhanced_content.endswith('.') else ' '
                    enhanced_content += f"{separator}{prefix}{additional_text}"
                    successful_concatenations += 1
                else:
                    logger.warning(f"Column '{column_name}' is empty for item: {item.get('title', 'unknown')}")
            else:
                logger.warning(f"Column '{column_name}' not found in item: {item.get('title', 'unknown')}")
                    
        
        # Update the chunking column with enhanced content
        enhanced_item[chunking_column] = enhanced_content
        enhanced_data.append(enhanced_item)
    
    logger.info(f"Value missing chunking column '{chunking_column}': {chunking_column_no_data}")
    logger.info(f"Enhanced {len(enhanced_data)} items with {successful_concatenations} successful concatenations")
    logger.info(f"Concatenated columns {[config.get('column') for config in columns_to_add]} to '{chunking_column}'")

    return enhanced_data
