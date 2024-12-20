import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from land_grab_2.stl_dataset.step_1.constants import GIS_ACRES, WGS_84

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

not_a_cession = [
    "336",
    "368",
    "496",
    "510",
    "524",
    "525",
    "552",
    "591",
    "596",
    "599",
    "618",
    "621",
    "625",
    "632",
    "633",
    "651",
    "702",
    "713",
    "718",
    "540a",
]
unknown_cession_data = ["717"]


def get_price_paid_per_acre(cession: str, price_info: str | None) -> float:
    """
    Get the price paid per acre for a cession.

    Arguments:
    cession -- The cession number, as a string.
    price_info -- The price information for the cession.

    Returns:
    float -- The price paid per acre for the cession.
    """
    if cession in not_a_cession:
        return "N/A"

    if cession in unknown_cession_data:
        return "Unknown"

    if not isinstance(price_info, str) and np.isnan(price_info):
        return "Unknown"

    price_info = price_info.lstrip("$")
    return 0.0 if not price_info else float(price_info)


def add_price_columns(
    stl_gdf: gpd.GeoDataFrame, cessions_price_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Add cession purchase price columns to the STL GeoDataFrame.

    Arguments:
    stl_gdf -- GeoDataFrame containing STL parcel geometries
    cessions_price_df -- DataFrame containing cession purchase price information

    Returns:
    GeoDataFrame -- STL parcels with cession purchase price information
    """
    log.info("Processing cession purchase price information.")
    cols = stl_gdf.columns.tolist()
    cession_cols = [c for c in cols if c.startswith("cession_num_")]
    cession_price_cols = [
        f"C{i}_price_paid_per_acre" for i, _ in enumerate(cession_cols, start=1)
    ]

    out_rows = []
    cession_prices_simple = {}
    stl_gdf_dict = stl_gdf.to_dict(orient="records")

    for i, row in enumerate(stl_gdf_dict):
        # Ensure all cessions have a corresponding, initially blank, price column.
        for col in cession_price_cols:
            row[col] = ""

        cession_nums = row["all_cession_numbers"].split(" ")
        cession_prices = {}

        for i, cession in enumerate(cession_nums, start=1):
            if cession in cession_prices_simple:
                row[f"C{i}_price_paid_per_acre"] = cession_prices_simple[cession]
                if isinstance(row[f"C{i}_price_paid_per_acre"], str):
                    continue
                cession_prices[(i, cession)] = row[f"C{i}_price_paid_per_acre"]
                continue

            cession_price_rows = cessions_price_df[
                cessions_price_df["Cession_Number"] == cession
            ].to_dict(orient="records")
            if not cession_price_rows:
                continue

            price_info = cession_price_rows[0]["US_Paid_Per_Acre - Inflation Adjusted"]
            row[f"C{i}_price_paid_per_acre"] = get_price_paid_per_acre(
                cession, price_info
            )
            cession_prices_simple[cession] = row[f"C{i}_price_paid_per_acre"]

            if isinstance(row[f"C{i}_price_paid_per_acre"], str):
                continue

            cession_prices[(i, cession)] = row[f"C{i}_price_paid_per_acre"]

        if GIS_ACRES in row and row[GIS_ACRES]:
            parcel_size = float(row[GIS_ACRES])
        else:
            parcel_size = 0.0

        row["price_paid_for_parcel"] = round(
            sum([price * parcel_size for price in cession_prices.values()]), 2
        )

        out_rows.append(row)

    col_seq = []
    cession_price_cols = cession_price_cols.copy()
    for col in cols:
        if "all_cession_numbers" in col:  # Direct check for the column name
            col_seq.append(col)
            col_seq.append('price_paid_for_parcel')
        elif col.endswith("present_day_tribe"):
            col_seq.append(cession_price_cols.pop(0))
            col_seq.append(col)
        else:
            col_seq.append(col)

    gdf = gpd.GeoDataFrame(out_rows, geometry=stl_gdf.geometry, crs=stl_gdf.crs)
    gdf = gdf[col_seq]

    return gdf


def run():
    print("Running Step 3: Calculate cession purchase price.")
    required_envs = ["DATA"]
    missing_envs = [env for env in required_envs if os.environ.get(env) is None]
    if any(missing_envs):
        raise Exception(
            f"RequiredEnvVar: The following ENV vars must be set. {missing_envs}"
        )

    data_tld = os.environ.get("DATA")
    prev_out_dir = Path(f"{data_tld}/stl_dataset/step_2_5/output").resolve()
    in_dir = Path(f"{data_tld}/stl_dataset/step_3/input").resolve()
    out_dir = Path(f"{data_tld}/stl_dataset/step_3/output").resolve()

    # Load parcels and cession prices.
    stl_gdf = gpd.read_file(
        prev_out_dir / "stl_dataset_extra_activities_plus_cessions.geojson"
    )
    cessions_price_df = pd.read_csv(in_dir / "Cession_Data.csv")

    # Add cession purchase price columns to the GeoDataFrame.
    stl_with_cession_price_gdf = add_price_columns(
        stl_gdf,
        cessions_price_df,
    )

    # Export the GeoDataFrame.
    log.info(
        "Writing output files to data/stl_dataset/step_3/output/stl_dataset_extra_activities_plus_cessions_plus_prices{_wgs84}.{csv,geojson}"
    )
    stl_with_cession_price_gdf.to_file(
        out_dir / "stl_dataset_extra_activities_plus_cessions_plus_prices.geojson",
        driver="GeoJSON",
    )

    # Export a WGS84 version of the GeoDataFrame.
    stl_with_cession_price_gdf.to_crs(WGS_84).to_file(
        out_dir
        / "stl_dataset_extra_activities_plus_cessions_plus_prices_wgs84.geojson",
        driver="GeoJSON",
    )

    # Export a CSV version of the GeoDataFrame.
    stl_with_cession_price_gdf.drop(columns=["geometry"], inplace=True)
    stl_with_cession_price_gdf.to_csv(
        out_dir / "stl_dataset_extra_activities_plus_cessions_plus_prices.csv",
        index=False,
    )


if __name__ == "__main__":
    run()
