import pandas as pd
import requests
import json
from urllib.parse import quote
import sys

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
    
    # Remove columns where all values are NaN
    df = df.dropna(axis=1, how='all')
    
    # Convert back to list of dictionaries
    return df.to_dict('records')

def main():
    # Read the input CSV
    try:
        df = pd.read_csv('SD-surface-complete.csv')
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)

    base_url = "https://sdgis.sd.gov/arcgis1/rest/services/SD_All/Boundary_PLSS_QuarterQuarter/MapServer/0"
    
    # Initialize lists to store features and matched data
    features = []
    matched_data = []
    
    total_records = len(df)
    match_count = 0
    
    print(f"Processing {total_records} records...")
    
    # Process each record
    for idx, row in df.iterrows():
        print(f"Processing record {idx + 1}/{total_records}: {row['SECDIVID']}", end='\r')
        
        # Query ArcGIS
        result = query_arcgis_feature(row['SECDIVID'], base_url)
        matches_found = False
        
        if result and 'features' in result and result['features']:
            # Process all matches
            for feature in result['features']:
                match_count += 1
                matches_found = True
                
                # Clean attributes before merging
                cleaned_attributes = {k: v for k, v in feature['attributes'].items() if v is not None}
                
                # Store GeoJSON feature
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
                
                # Store matched data for CSV
                matched_row = {
                    **row.to_dict(),
                    **cleaned_attributes
                }
                matched_data.append(matched_row)
        
        if not matches_found:
            # If no match, keep original data for CSV (left join)
            matched_data.append(row.to_dict())
    
    print(f"\nProcessing complete. Found {match_count} matches across {total_records} source records.")
    print("Cleaning and exporting results...")
    
    # Clean the matched data to remove empty columns
    cleaned_matched_data = clean_data(matched_data)
    
    # Clean the GeoJSON features
    for feature in features:
        feature['properties'] = {k: v for k, v in feature['properties'].items() if pd.notna(v)}
    
    # Create GeoJSON structure
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Export GeoJSON
    try:
        with open('SD-surface.geojson', 'w') as f:
            json.dump(geojson, f)
        print("GeoJSON exported successfully to SD-surface.geojson")
    except Exception as e:
        print(f"Error exporting GeoJSON: {str(e)}")
    
    # Export CSV
    try:
        cleaned_df = pd.DataFrame(cleaned_matched_data)
        cleaned_df.to_csv('SD-surface.csv', index=False)
        print("CSV exported successfully to SD-surface.csv")
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")

if __name__ == "__main__":
    main()
