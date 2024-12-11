import logging
import os
from pathlib import Path
from typing import List
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np

from land_grab_2.stl_dataset.step_1.constants import WGS_84

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def area_join(
    source_df: gpd.GeoDataFrame, target_df: gpd.GeoDataFrame, variables: List[str]
) -> gpd.GeoDataFrame:
    """
    Join variables from source_df based on the largest intersection with feat-
    ures from target_df. In case of a tie, pick the first joined record.
    Implementation adapted from tobler:
    https://pysal.org/tobler/_modules/tobler/area_weighted/area_join.html#area_join

    Arguments:
    source_df -- the GeoDataFrame containing source values
    target_df -- the GeoDataFrame containing target values
    variables -- string or list-like column(s) in source_df dataframe to join to
                 target_df

    Returns:
    geopandas.GeoDataFrame -- target_df GeoDataFrame with joined variables as
                              additional columns

    """
    for v in variables:
        if v in target_df.columns:
            raise ValueError(f"Column '{v}' already present in target_df.")

    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query(
        target_df.geometry, predicate="intersects"
    )
    areas = (
        target_df.geometry.values[target_ix]
        .intersection(source_df.geometry.values[source_ix])
        .area
    )

    main = []
    for i in range(len(target_df)):
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)

    main = np.array(main, dtype=float)
    mask = ~np.isnan(main)

    for v in variables:
        arr = np.empty(len(main), dtype=object)
        arr[mask] = source_df[v].values[main[mask].astype(int)]
        try:
            arr = arr.astype(source_df[v].dtype)
        except TypeError:
            warnings.warn(
                f"Cannot preserve dtype of '{v}'. Falling back to `dtype=object`.",
            )
        target_df[v] = arr

    return target_df


def join_counties_to_parcels(
    parcels_gdf: gpd.GeoDataFrame, counties_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Join county name to parcels based on the largest spatial intersection.

    Arguments:
    parcels_gdf -- GeoDataFrame containing parcel geometries
    counties_gdf -- GeoDataFrame containing county geometries

    Returns:
    GeoDataFrame -- STL parcels with county names joined as a new column
    """
    # Reproject counties_gdf to match parcels_gdf.
    counties_gdf = counties_gdf.to_crs(parcels_gdf.crs)

    out_gdf = area_join(
        counties_gdf,
        parcels_gdf,
        variables=["NAME"],
    )

    # Drop the existing county column—we'll use the result of this join instead.
    out_gdf = out_gdf.drop(columns=["county"])
    out_gdf = out_gdf.rename(columns={"NAME": "county"})

    return out_gdf


def join_cessions_to_parcels(
    parcels_gdf: gpd.GeoDataFrame, cessions_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Join parcel attributes to cessions based on all spatial intersections.

    This is a one-to-many left join; a given parcel may intersect many (up to 8)
    cessions.

    Arguments:
    parcels_gdf -- GeoDataFrame containing parcel geometries
    cessions_gdf -- GeoDataFrame containing cession geometries

    Returns:
    GeoDataFrame -- STL parcels with each parcel-cession intersection as a dis-
                    tinct row
    """
    # Reproject cessions_gdf to match parcels_gdf.
    cessions_gdf = cessions_gdf.to_crs(parcels_gdf.crs)

    # Subset cessions_gdf to only include the CESSNUM and geometry columns.
    cessions_gdf = cessions_gdf[["CESSNUM", "geometry"]]

    out_gdf = parcels_gdf.sjoin(cessions_gdf, how="left")
    out_gdf.drop(columns=["index_right"], inplace=True)

    return out_gdf


def aggregate_cessions_by_parcel(
    parcels_gdf: gpd.GeoDataFrame, cession_codebook_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate cession information for each parcel.

    This function aggregates all parcel-cession intersections into a single row,
    one per parcel. Cession numbers are space-separated in the all_cession_numb-
    ers field. Additionally, each parcel will have 8 cession number fields,
    8 present day tribe fields, and 8 historical tribe fields.

    Arguments:
    parcels_gdf -- GeoDataFrame containing parcel geometries
    cession_codebook_df -- DataFrame containing cession information

    Returns:
    GeoDataFrame -- STL parcels with cession information aggregated by parcel
    """

    def expand_cession_numbers(group):
        # Split the all_cession_numbers field into a list.
        cession_list = group["all_cession_numbers"].split()

        # Create fields for tracking individual cession numbers, present day
        # tribes, and tribes named in cessions.
        cession_fields = {}
        for i in range(8):
            cession_num = f"cession_num_{i+1:02d}"
            present_day_tribe = f"C{i+1:02d}_present_day_tribe"
            historical_tribe = f"C{i+1:02d}_tribe_named_in_land_cessions_1784-1894"

            cession_fields[cession_num] = (
                cession_list[i] if i < len(cession_list) else None
            )

            # Look up the present day and historical tribes for each cession
            # number, if available.
            if cession_fields[cession_num] is not None:
                matching_cession = cession_codebook_df.loc[
                    cession_codebook_df["Cession_Number"] == cession_fields[cession_num]
                ]

                if not matching_cession.empty:
                    cession_fields[present_day_tribe] = matching_cession.iloc[0][
                        "Present_Day_Tribe"
                    ]
                    cession_fields[historical_tribe] = matching_cession.iloc[0][
                        "Tribe_Named_in_Land_Cessions_1784-1894"
                    ]
                else:
                    cession_fields[present_day_tribe] = None
                    cession_fields[historical_tribe] = None
            else:
                cession_fields[present_day_tribe] = None
                cession_fields[historical_tribe] = None

        # Combine the new cession fields with the original group data.
        return pd.Series({**group.to_dict(), **cession_fields})

    crs = parcels_gdf.crs

    out_gdf = (
        parcels_gdf.groupby("object_id")
        .agg(
            {
                **{
                    col: "first"
                    for col in parcels_gdf.columns
                    if col != "object_id" and col != "CESSNUM"
                },
                "CESSNUM": lambda x: " ".join(map(str, x.unique())),
            }
        )
        .rename(columns={"CESSNUM": "all_cession_numbers"})
        .reset_index()
        .apply(expand_cession_numbers, axis=1)
    )

    return gpd.GeoDataFrame(out_gdf, geometry=out_gdf["geometry"], crs=crs)


def run():
    print("Running Step 2.5: Join cession and county information to parcels.")
    required_envs = ["DATA"]
    missing_envs = [env for env in required_envs if os.environ.get(env) is None]
    if any(missing_envs):
        raise Exception(
            f"RequiredEnvVar: The following ENV vars must be set. {missing_envs}"
        )

    data_tld = os.environ.get("DATA")
    in_dir = Path(f"{data_tld}/stl_dataset/step_2_5/input").resolve()
    out_dir = Path(f"{data_tld}/stl_dataset/step_2_5/output").resolve()

    # Load parcels, counties, and cessions data.
    parcels_gdf = gpd.read_file(
        Path(
            f"{data_tld}/stl_dataset/step_2/output/stl_dataset_extra_activities.geojson"
        ).resolve()
    )
    counties_gdf = gpd.read_file(in_dir / "us_counties.json")
    cessions_gdf = gpd.read_file(in_dir / "cessions.geojson")

    # Additionally, load the cession codebook.
    cession_codebook_df = pd.read_csv(in_dir / "cession-codebook.csv")

    # Join county information to parcels.
    log.info("Joining county information to parcels.")
    parcels_counties_gdf = join_counties_to_parcels(parcels_gdf, counties_gdf)

    # Join cession information to parcels.
    log.info("Joining cession information to parcels.")
    parcels_counties_cessions_gdf = join_cessions_to_parcels(
        parcels_counties_gdf, cessions_gdf
    )

    # Aggregate cessions by parcel.
    log.info("Aggregating each parcel's cessions.")
    parcels_cessions_gdf = aggregate_cessions_by_parcel(
        parcels_counties_cessions_gdf, cession_codebook_df
    )

    # Export the GeoDataFrame.
    log.info(
        "Writing output files to data/stl_dataset/step_2_5/output/stl_dataset_extra_activities_plus_cessions{_wgs84}.{csv,geojson}"
    )
    parcels_cessions_gdf.to_file(
        out_dir / "stl_dataset_extra_activities_plus_cessions.geojson", driver="GeoJSON"
    )

    # Export a WGS84 version of the GeoDataFrame.
    parcels_cessions_gdf.to_crs(WGS_84).to_file(
        out_dir / "stl_dataset_extra_activities_plus_cessions_wgs84.geojson",
        driver="GeoJSON",
    )

    # Export a CSV version of the GeoDataFrame.
    parcels_cessions_gdf.drop(columns=["geometry"], inplace=True)
    parcels_cessions_gdf.to_csv(
        out_dir / "stl_dataset_extra_activities_plus_cessions.csv", index=False
    )


if __name__ == "__main__":
    run()
