# filepath: /Users/sajjadhossain/Documents/01 - Projects/05 MSc/nlp-project/src/specialization/data/processed/to_csv.py

import json
import csv
import os
from typing import Any

def clean_and_flatten_value(value: Any) -> str:
    """
    Convert any value to a clean string suitable for CSV.
    Handles None, lists, and special characters properly.
    """
    if value is None:
        return ""
    
    if isinstance(value, list):
        # Join list items with semicolon separator
        # Clean each item to handle nested structures and special chars
        cleaned_items = []
        for item in value:
            if isinstance(item, (dict, list)):
                cleaned_items.append(str(item).replace('"', '""'))
            else:
                cleaned_items.append(str(item).replace('"', '""'))
        return "; ".join(cleaned_items)
    
    if isinstance(value, dict):
        # Convert dict to string representation
        return str(value).replace('"', '""')
    
    # Convert to string and handle special characters
    str_value = str(value)
    
    # Handle quotes by doubling them (CSV standard)
    str_value = str_value.replace('"', '""')
    
    # Handle newlines and carriage returns
    str_value = str_value.replace('\n', ' ').replace('\r', ' ')
    
    # Handle other potential problematic characters
    str_value = str_value.replace('\t', ' ')
    
    return str_value

def json_to_csv(json_file_path: str) -> None:
    """
    Convert JSON file to CSV with the same name in the same directory.
    Handles all data types and special characters robustly.
    """
    # Validate input file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # Create output CSV file path
    base_name = os.path.splitext(json_file_path)[0]
    csv_file_path = f"{base_name}.csv"
    
    print(f"Loading JSON from: {json_file_path}")
    
    # Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error reading JSON: {e}")
    
    if not data:
        raise ValueError("JSON file is empty or contains no data")
    
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of objects")
    
    print(f"Found {len(data)} records")
    
    # Get all unique keys from all records (in case some records have different fields)
    all_keys = set()
    for record in data:
        if isinstance(record, dict):
            all_keys.update(record.keys())
    
    # Sort keys for consistent column order
    fieldnames = sorted(list(all_keys))
    
    print(f"Writing CSV to: {csv_file_path}")
    print(f"Columns: {fieldnames}")
    
    # Write CSV file
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,  # Quote all fields for maximum safety
                escapechar='\\',
                extrasaction='ignore'
            )
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for i, record in enumerate(data):
                if not isinstance(record, dict):
                    print(f"Warning: Skipping non-dict record at index {i}")
                    continue
                
                # Clean and prepare row data
                clean_record = {}
                for field in fieldnames:
                    raw_value = record.get(field)
                    clean_record[field] = clean_and_flatten_value(raw_value)
                
                writer.writerow(clean_record)
                
                # Progress indicator for large files
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1} records...")
    
    except IOError as e:
        raise IOError(f"Error writing CSV file: {e}")
    
    print(f"Successfully converted {len(data)} records to CSV")
    print(f"Output file: {csv_file_path}")

def main():
    """Main function to run the conversion."""
    # Path to the JSON file
    json_file_path = "/Users/sajjadhossain/Documents/01 - Projects/05 MSc/nlp-project/src/specialization/data/processed/processed_movies_data_sample.json"
    
    try:
        json_to_csv(json_file_path)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())