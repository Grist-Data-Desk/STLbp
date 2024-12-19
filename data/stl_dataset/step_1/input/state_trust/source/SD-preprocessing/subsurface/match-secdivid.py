import pandas as pd
import requests
import json
import sys
import argparse
from pathlib import Path

def query_arcgis_feature(secdivid, base_url):
    """Query ArcGIS REST service for a specific SECDIVID"""
    query_url = f"{base_url}/query"
    
    params = {
        'where': f"SECDIVID = '{secdivid}'",
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'json',
        'outSR': '4326'  # WGS 84
    }
    
    try:
        response = requests.get(query_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying SECDIVID {secdivid}: {str(e)}")
        return None

def clean_data(data_list):
    """Convert list of dictionaries to DataFrame and remove empty columns"""
    df = pd.DataFrame(data_list)
    df = df.dropna(axis=1, how='all')
    return df.to_dict('records')

def process_data(input_file, output_prefix=None, base_url=None):
    """Process the input CSV file and generate output files"""
    # Set default base URL if not provided
    if base_url is None:
        base_url = "https://sdgis.sd.gov/arcgis1/rest/services/SD_All/Boundary_PLSS_QuarterQuarter/MapServer/0"
    
    # Set output prefix to input filename if not provided
    if output_prefix is None:
        output_prefix = Path(input_file).stem
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)
    
    if 'SECDIVID' not in df.columns:
        print("Error: Input CSV must contain a 'SECDIVID' column")
        sys.exit(1)
    
    features = []
    matched_data = []
    total_records = len(df)
    match_count = 0
    
    print(f"Processing {total_records} records...")
    
    for idx, row in df.iterrows():
        print(f"Processing record {idx + 1}/{total_records}: {row['SECDIVID']}", end='\r')
        
        result = query_arcgis_feature(row['SECDIVID'], base_url)
        matches_found = False
        
        if result and 'features' in result and result['features']:
            for feature in result['features']:
                match_count += 1
                matches_found = True
                
                cleaned_attributes = {k: v for k, v in feature['attributes'].items() if v is not None}
                
                geojson_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': feature['geometry']['rings']
                    },
                    'properties': {
                        **row.to_dict(),
                        **cleaned_attributes
                    }
                }
                features.append(geojson_feature)
                
                matched_row = {
                    **row.to_dict(),
                    **cleaned_attributes
                }
                matched_data.append(matched_row)
        
        if not matches_found:
            matched_data.append(row.to_dict())
    
    print(f"\nProcessing complete. Found {match_count} matches across {total_records} source records.")
    print("Cleaning and exporting results...")
    
    cleaned_matched_data = clean_data(matched_data)
    
    for feature in features:
        feature['properties'] = {k: v for k, v in feature['properties'].items() if pd.notna(v)}
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Export files with prefix
    geojson_file = f"{output_prefix}.geojson"
    csv_file = f"{output_prefix}.csv"
    
    try:
        with open(geojson_file, 'w') as f:
            json.dump(geojson, f)
        print(f"GeoJSON exported successfully to {geojson_file}")
    except Exception as e:
        print(f"Error exporting GeoJSON: {str(e)}")
    
    try:
        cleaned_df = pd.DataFrame(cleaned_matched_data)
        cleaned_df.to_csv(csv_file, index=False)
        print(f"CSV exported successfully to {csv_file}")
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process PLSS data from CSV and generate GeoJSON/CSV outputs')
    parser.add_argument('input_file', help='Input CSV file containing SECDIVID column')
    parser.add_argument('--output-prefix', '-o', help='Prefix for output files (default: input filename)')
    parser.add_argument('--base-url', '-u', help='Base URL for ArcGIS REST service')
    
    args = parser.parse_args()
    
    process_data(args.input_file, args.output_prefix, args.base_url)

if __name__ == "__main__":
    main()
