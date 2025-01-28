import itertools
import os
import traceback
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from land_grab_2.stl_dataset.step_1.constants import GIS_ACRES, STATE, TRIBE_SUMMARY, \
    RIGHTS_TYPE
from land_grab_2.utilities.utils import prettyify_list_of_strings

os.environ['RESTAPI_USE_ARCPY'] = 'FALSE'

TRIBE_COMBINE_DETAILS = {
    'Bridgeport Indian Colony, California': 'Bridgeport Paiute Indian Colony of California',
    'Burns Paiute Tribe, Oregon': 'Burns Paiute Tribe of the Burns Paiute Indian Colony of Oregon',
    'Confederated Tribes and Bands of the Yakama Nation': 'Confederated Tribes and Bands of the Yakama Nation, Washington',
    'Nez Perce Tribe, Idaho': 'Nez Perce Tribe of Idaho',
    'Quinault Indian Nation, Washington': 'Quinault Tribe of the Quinault Reservation, Washington',
    'Confederated Tribes of the Umatilla Reservation, Oregon': 'Confederated Tribes of the Umatilla Indian Reservation, Oregon',
    'Shoshone-Bannock Tribes of the Fort Hall Reservation, Idaho': 'Shoshone-Bannock Tribes of the Fort Hall Reservation of Idaho'
}

def dedup_tribe_names(tribe_list):
    tribe_list_deduped = list(sorted(set([t
                                          if t not in TRIBE_COMBINE_DETAILS else TRIBE_COMBINE_DETAILS[t]
                                          for t in tribe_list])))
    return tribe_list_deduped

def extract_tribe_list(tribe_list, should_join=True):
    if tribe_list is None:
        return '' if should_join else []

    raw_tribes = [
        x
        for x in list(itertools.chain.from_iterable([
            '' if not isinstance(i, str) and pd.isna(i) else i.split(';')
            for i in tribe_list
        ])) if x]
    tribe_list = list(sorted(set([i.strip() for i in raw_tribes])))
    tribe_list = dedup_tribe_names(tribe_list)
    if should_join:
        tribe_list = ';'.join(tribe_list)
    return tribe_list

def cleanup_gis_acres(row):
    gis_acres = GIS_ACRES if GIS_ACRES in row.keys() else 'acres'
    original_val = row[gis_acres]
    if isinstance(original_val, str) and len(original_val) == 0:
        return None

    if isinstance(original_val, str) and len(original_val) > 0:
        return float(original_val)

    return original_val

def gather_single_tribe_details(row, current_tribe, cession_number_col):
    try:
        gis_acres_val = cleanup_gis_acres(row)
        current_tribe = (current_tribe
                         if current_tribe not in TRIBE_COMBINE_DETAILS
                         else TRIBE_COMBINE_DETAILS[current_tribe])
        return {
            GIS_ACRES: gis_acres_val,
            'present_day_tribe': current_tribe,
            RIGHTS_TYPE: row[RIGHTS_TYPE],
            STATE: row[STATE],
            'cession_number': row[cession_number_col],
            'geometry': row['geometry'],
        }
    except Exception as err:
        print(traceback.format_exc())
        print(err)

def construct_single_tribe_info(row):
    try:
        present_day_tribe_cols = [c for c in row.keys() if 'present_day_tribe' in c]
        tribe_cession_number_cols = [c for c in row.keys() if 'cession_num' in c and 'all' not in c]

        tribe_records = []
        for present_day_tribe_col, cession_number_col in zip(present_day_tribe_cols, tribe_cession_number_cols):
            current_tribe = row[present_day_tribe_col]
            if not isinstance(current_tribe, str) or (isinstance(current_tribe, str) and len(current_tribe) == 0):
                continue

            records = [
                gather_single_tribe_details(row, t.strip(), cession_number_col)
                for t in extract_tribe_list(current_tribe.split(';'), should_join=False)
            ]
            tribe_records += [r for r in records if r is not None]

        return tribe_records
    except Exception as err:
        print(err)

def gis_acres_sum_by_rights_type_tribe_summary(df):
    tribe_land_accounts = defaultdict(dict)
    for row in df.to_dict(orient='records'):
        tribe = row['present_day_tribe']

        rights_type = 'unknown_rights_type' if RIGHTS_TYPE not in row or not row[RIGHTS_TYPE] else row[RIGHTS_TYPE]
        if '+' in rights_type:
            rights_type = rights_type.replace('+', '_and_')

        out_col_name = f'{rights_type}_acres'
        if out_col_name not in tribe_land_accounts[tribe]:
            tribe_land_accounts[tribe][out_col_name] = float(row[GIS_ACRES])
        else:
            tribe_land_accounts[tribe][out_col_name] += float(row[GIS_ACRES])

    records = [{'present_day_tribe': tribe, **land_info} for tribe, land_info in tribe_land_accounts.items()]

    return pd.DataFrame(records)

def tribe_summary(gdf, output_dir):
    results = list(
        itertools.chain.from_iterable(
            [construct_single_tribe_info(row.to_dict()) for _, row in gdf.iterrows()]
        )
    )
    parcels_by_tribe = gpd.GeoDataFrame(results, crs=gdf.crs)
    parcels_by_tribe.to_file(output_dir / "parcels-by-tribe.geojson", driver="GeoJSON")
    tribe_summary_tmp = pd.DataFrame(results).drop(columns=["geometry"])
    group_cols = [
        c
        for c in list(tribe_summary_tmp.columns)
        if GIS_ACRES not in c and "cession_number" not in c
    ]

    # Create the semi-aggregated tribe summary as both CSV and GeoJSON.
    tribe_summary_semi_aggd = (
        tribe_summary_tmp.groupby(group_cols)[GIS_ACRES].sum().reset_index()
    )
    tribe_summary_semi_aggd.to_csv(output_dir / TRIBE_SUMMARY)

    # Create the fully-aggregated tribe summary
    tribe_summary_full_agg = (
        tribe_summary_tmp.groupby(["present_day_tribe"]).agg(list).reset_index()
    )
    tribe_summary_full_agg[GIS_ACRES] = tribe_summary_full_agg[GIS_ACRES].map(sum)
    tribe_summary_full_agg = tribe_summary_full_agg.apply(
        prettyify_list_of_strings, axis=1
    )
    tribe_summary_full_agg["cession_count"] = tribe_summary_full_agg[
        "cession_number"
    ].map(lambda v: len(v.split(",")))
    tribe_summary_full_agg["cession_number"] = tribe_summary_full_agg[
        "cession_number"
    ].replace(".0", "")

    rights = gis_acres_sum_by_rights_type_tribe_summary(tribe_summary_tmp)
    tribe_summary_full_agg = tribe_summary_full_agg.join(
        rights.set_index("present_day_tribe"), on="present_day_tribe"
    )

    tribe_summary_full_agg["surface_acres"] = tribe_summary_full_agg[
        "surface_acres"
    ].map(lambda v: round(v, 2))
    tribe_summary_full_agg["subsurface_acres"] = tribe_summary_full_agg[
        "subsurface_acres"
    ].map(lambda v: round(v, 2))
    tribe_summary_full_agg["timber_acres"] = tribe_summary_full_agg["timber_acres"].map(
        lambda v: round(v, 2)
    )

    # Sequence columns in a particular order.
    tribe_summary_full_agg = tribe_summary_full_agg[
        [
            "present_day_tribe",
            "cession_count",
            "cession_number",
            *list(
                sorted(
                    c
                    for c in tribe_summary_full_agg.columns
                    if c.endswith("_acres") and GIS_ACRES not in c
                )
            ),
            "state",
        ]
    ]

    tribe_summary_full_agg.to_csv(output_dir / "tribe-summary-condensed.csv")

def calculate_summary_statistics_helper(summary_statistics_data_directory):
    '''
    Calculate summary statistics based on the full dataset. Creates a CSV for each present
    day tribe with total acreage of state land trust parcels, all associated cessions, and all
    states and universities that have land taken from this tribe held in trust
    '''

    data_tld = Path(os.environ.get('DATA')).resolve()
    input_file = data_tld / 'stl_dataset/step_3/output/stl_dataset_extra_activities_plus_cessions_plus_prices_wgs84.geojson'
    output_dir = data_tld / 'stl_dataset/step_4/output'
    gdf = gpd.read_file(input_file)

    gdf_tribes = gdf.copy(deep=True)

    stats_dir = Path(summary_statistics_data_directory).resolve()
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    gis_acres_col = GIS_ACRES if GIS_ACRES in gdf_tribes.columns else 'gis_calculated_acres'
    gdf_tribes[GIS_ACRES] = gdf_tribes[gis_acres_col].astype(float)

    tribe_summary(gdf_tribes, output_dir)
