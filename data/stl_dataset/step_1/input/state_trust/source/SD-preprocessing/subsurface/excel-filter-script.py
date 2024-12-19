import pandas as pd
import os
import re
from typing import List, Optional

def find_header_row(sheet: pd.DataFrame) -> int:
    """Find the row containing headers by looking for common column identifiers"""
    for idx, row in sheet.iterrows():
        row_values = [str(val).lower() for val in row if pd.notna(val)]
        if any('township' in val or 'range' in val or 'section' in val or 'description' in val for val in row_values):
            return idx
    return 0

def get_filtered_dataframe(df: pd.DataFrame, sheet_name: str, desc_col: str) -> Optional[pd.DataFrame]:
    """Filter dataframe and add county column"""
    if desc_col not in df.columns:
        print(f"Warning: Could not find description column '{desc_col}' in sheet {sheet_name}")
        return None
    
    # Forward fill blank values in Township and Range columns
    df['Township'] = df['Township'].ffill()
    df['Range'] = df['Range'].ffill()
    
    # Debug print some sample values before filtering
    print("\nSample values from description column before filtering:")
    sample_values = df[desc_col].dropna().sample(min(5, len(df))).tolist()
    for val in sample_values:
        print(f"'{val}'")
    
    # Replace various types of whitespace and normalize dashes
    df[desc_col] = df[desc_col].replace(to_replace=['\xa0', '\t', '\n', '\r', '\u2013', '\u2014', '–', '—'], value=' ', regex=True)
    df[desc_col] = df[desc_col].str.strip()
    
    # More flexible pattern that handles various dash types and spacing
    pattern = r'.*[-–—]\s*T$'
    
    # Print values that end with any kind of dash followed by T
    print("\nValues ending with dash and T:")
    matching_values = df[df[desc_col].str.match(pattern, na=False)][desc_col]
    for val in matching_values:
        print(f"'{val}'")
    
    filtered_df = df[df[desc_col].str.match(pattern, na=False)].copy()
    
    if filtered_df.empty:
        print(f"No rows found ending with '-T' pattern in sheet {sheet_name}")
        return None
    
    # Add county (sheet name) column
    filtered_df['County'] = sheet_name
    print(f"Found {len(filtered_df)} rows ending with '-T' pattern in sheet {sheet_name}")
    
    # Debug print the matched rows
    print("\nMatched descriptions:")
    for val in filtered_df[desc_col]:
        print(f"'{val}'")
    
    return filtered_df

def process_excel_file(file_path: str) -> pd.DataFrame:
    """Process Excel file and return combined filtered data"""
    excel_file = pd.ExcelFile(file_path)
    filtered_dfs: List[pd.DataFrame] = []
    
    for sheet_name in excel_file.sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        
        # First read without headers to find header row
        temp_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        header_row = find_header_row(temp_df)
        
        # Read sheet again with correct header row
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        
        # Make sure Township and Range columns exist
        expected_columns = ['Township', 'Range']
        if not all(col in df.columns for col in expected_columns):
            print(f"Warning: Missing required columns in sheet {sheet_name}")
            print("Found columns:", df.columns.tolist())
            continue
        
        # Find the description column
        desc_col = None
        for col in df.columns:
            if 'description' in str(col).lower() or 'class' in str(col).lower():
                desc_col = col
                break
        
        if desc_col is None:
            print(f"Warning: Could not find description column in sheet {sheet_name}")
            continue
        
        filtered_df = get_filtered_dataframe(df, sheet_name, desc_col)
        if filtered_df is not None:
            filtered_dfs.append(filtered_df)
    
    if not filtered_dfs:
        print("No matching data found in any sheet")
        return pd.DataFrame()
    
    # Ensure all DataFrames have the same columns before concatenation
    all_columns = set().union(*(df.columns for df in filtered_dfs))
    for df in filtered_dfs:
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = pd.NA
    
    try:
        result_df = pd.concat(filtered_dfs, ignore_index=True)
        
        # Print final results
        print("\nFinal matched descriptions:")
        for val in result_df[desc_col]:
            print(f"'{val}'")
            
        return result_df
    except Exception as e:
        print(f"Error combining data: {str(e)}")
        return pd.DataFrame()

def main():
    file_name = "SD Electronic Mineral  Book.xlsx"
    if not os.path.exists(file_name):
        print(f"Error: Could not find file '{file_name}' in current directory")
        return
    
    print(f"Processing file: {file_name}")
    
    try:
        result_df = process_excel_file(file_name)
        
        if not result_df.empty:
            output_path = f'filtered_{os.path.splitext(file_name)[0]}.csv'
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            print(f"Found total of {len(result_df)} matching rows")
        else:
            print("No matching rows found in file")
            
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    main()
