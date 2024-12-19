#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
import argparse
import sys
from pathlib import Path
import numpy as np

def get_utm_epsg(longitude, latitude):
    """
    Calculate the EPSG code for the UTM zone based on lat/lon.
    Returns EPSG code for appropriate UTM zone.
    """
    try:
        # Handle NaN values
        if pd.isna(longitude) or pd.isna(latitude):
            return None
            
        zone_number = int((longitude + 180) / 6) + 1
        
        # North or South hemisphere
        epsg = 26900 + zone_number if latitude > 0 else 32700 + zone_number
        
        return epsg
    except Exception as e:
        print(f"Error calculating UTM zone for lon: {longitude}, lat: {latitude}")
        return None

def filter_points_within_polygons(points_file, polygons_file, output_csv, output_geojson, 
                                lat_col='Latitude', lon_col='Longitude', buffer_feet=500):
    """
    Filter points that fall within polygons (with buffer) and export results.
    Uses appropriate UTM zones for accurate distance calculations.
    """
    try:
        # Read the input files
        print(f"Reading points from {points_file}...")
        points_df = pd.read_csv(points_file)
        
        # Check if latitude and longitude columns exist
        if lat_col not in points_df.columns or lon_col not in points_df.columns:
            raise ValueError(f"Could not find {lat_col} and/or {lon_col} columns in the CSV file")
        
        points_gdf = gpd.GeoDataFrame(
            points_df, 
            geometry=gpd.points_from_xy(points_df[lon_col], points_df[lat_col]),
            crs="EPSG:4326"  # WGS 84
        )
        
        print(f"Reading polygons from {polygons_file}...")
        polygons_gdf = gpd.read_file(polygons_file)
        
        # Ensure both layers are in WGS 84
        if polygons_gdf.crs != "EPSG:4326":
            print("Converting polygons to WGS 84...")
            polygons_gdf = polygons_gdf.to_crs("EPSG:4326")
        
        # Remove any invalid or empty geometries
        polygons_gdf = polygons_gdf[~polygons_gdf.geometry.is_empty & polygons_gdf.geometry.is_valid]
        
        print(f"Creating {buffer_feet} foot buffer around polygons...")
        buffered_polygons = []
        
        # Use bounds of each polygon to determine UTM zone
        for idx, polygon in polygons_gdf.iterrows():
            if polygon.geometry is None:
                continue
                
            # Get the centroid coordinates
            try:
                centroid = polygon.geometry.centroid
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                
                if utm_epsg is None:
                    continue
                    
                # Project single polygon to UTM
                poly_utm = gpd.GeoDataFrame(geometry=[polygon.geometry], crs="EPSG:4326")
                poly_utm = poly_utm.to_crs(f"EPSG:{utm_epsg}")
                
                # Buffer in feet
                poly_buffered = poly_utm.geometry.buffer(buffer_feet)
                
                # Convert back to WGS 84
                poly_wgs84 = gpd.GeoDataFrame(
                    geometry=poly_buffered,
                    crs=f"EPSG:{utm_epsg}"
                ).to_crs("EPSG:4326")
                
                buffered_polygons.append(poly_wgs84)
                
            except Exception as e:
                print(f"Warning: Error processing polygon {idx}: {str(e)}")
                continue
        
        if not buffered_polygons:
            raise ValueError("No valid buffered polygons were created")
        
        # Combine all buffered polygons
        all_buffered = gpd.GeoDataFrame(
            geometry=pd.concat(buffered_polygons).geometry,
            crs="EPSG:4326"
        )
        
        print("Filtering points within buffered polygons...")
        # Perform spatial join to get points within buffered polygons
        filtered_points = gpd.sjoin(
            points_gdf,
            all_buffered,
            how="inner",
            predicate="within"
        )
        
        # Drop the index_right column that was added by the spatial join
        if 'index_right' in filtered_points.columns:
            filtered_points = filtered_points.drop(columns=['index_right'])
        
        # Create output directory if it doesn't exist
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_geojson).parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV (without geometry column)
        print(f"Saving CSV to {output_csv}...")
        filtered_points.drop(columns=['geometry']).to_csv(output_csv, index=False)
        
        # Export to GeoJSON
        print(f"Saving GeoJSON to {output_geojson}...")
        filtered_points.to_file(output_geojson, driver='GeoJSON')
        
        print(f"\nResults:")
        print(f"Original points count: {len(points_gdf)}")
        print(f"Filtered points count: {len(filtered_points)}")
        print(f"Filtered out {len(points_gdf) - len(filtered_points)} points")
        print(f"Buffer distance used: {buffer_feet} feet")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Filter points within polygons (with buffer) and export results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('points_file', 
                       help='Input CSV file containing point data')
    parser.add_argument('polygons_file', 
                       help='Input GeoJSON file containing polygon data')
    parser.add_argument('--output-csv', 
                       default='filtered_points.csv',
                       help='Output CSV file path')
    parser.add_argument('--output-geojson', 
                       default='filtered_points.geojson',
                       help='Output GeoJSON file path')
    parser.add_argument('--lat-col', 
                       default='Latitude',
                       help='Name of latitude column in CSV')
    parser.add_argument('--lon-col', 
                       default='Longitude',
                       help='Name of longitude column in CSV')
    parser.add_argument('--buffer', 
                       type=float,
                       default=500,
                       help='Buffer distance in feet')
    
    args = parser.parse_args()
    
    filter_points_within_polygons(
        args.points_file,
        args.polygons_file,
        args.output_csv,
        args.output_geojson,
        args.lat_col,
        args.lon_col,
        args.buffer
    )

if __name__ == '__main__':
    main()
