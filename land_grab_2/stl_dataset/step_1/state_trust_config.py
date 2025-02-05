from land_grab_2.stl_dataset.step_1.constants import (
    DOWNLOAD_TYPE,
    API_QUERY_DOWNLOAD_TYPE,
    SHAPEFILE_DOWNLOAD_TYPE,
    STATE,
    UNIVERSITY,
    MANAGING_AGENCY,
    DATA_SOURCE,
    LOCAL_DATA_SOURCE,
    ATTRIBUTE_LABEL_TO_FILTER_BY,
    ATTRIBUTE_CODE_TO_ALIAS_MAP,
    RIGHTS_TYPE,
    SURFACE_RIGHTS_TYPE,
    SUBSURFACE_RIGHTS_TYPE,
    TRUST_NAME,
    TIMBER_RIGHTS_TYPE,
    LAYER,
    STATE_TRUST_DATA_SOURCE_DIRECTORY,
    EXISTING_COLUMN_TO_FINAL_COLUMN_MAP,
    ACRES,
    COUNTY,
    MERIDIAN,
    TOWNSHIP,
    RANGE,
    SECTION,
    ALIQUOT,
    BLOCK,
    ACTIVITY,
    STATE_ENABLING_ACT,
    NET_ACRES,
    GEOJSON_TYPE,
)

STATE_TRUST_CONFIGS = {
    "AZ-surface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "AZ",
        UNIVERSITY: "University of Arizona",
        MANAGING_AGENCY: "State Land Department",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "36 Stat. 557-579 (1910)",
        DATA_SOURCE: "https://server.azgeo.az.gov/arcgis/rest/services/azland/State_Trust_Parcels/MapServer/0",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["fundtxt"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'PENITENTIARY LAND FUND'": "PENITENTIARY LAND FUND",
            "'ST CHRTBL, PENAL & REFORM INST'": "ST CHRTBL, PENAL & REFORM INST",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "acres": ACRES,
            "County": COUNTY,
        },
    },
    "AZ-subsurface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "AZ",
        UNIVERSITY: "University of Arizona",
        MANAGING_AGENCY: "State Land Department",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "36 Stat. 557-579 (1910)",
        DATA_SOURCE: "https://server.azgeo.az.gov/arcgis/rest/services/azland/Mineral_Parcels/MapServer/0",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["fundtxt"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'PENITENTIARY LAND FUND'": "PENITENTIARY LAND FUND",
            "'ST CHRTBL, PENAL & REFORM INST'": "ST CHRTBL, PENAL & REFORM INST",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "acres": ACRES,
            "County": COUNTY,
        },
    },
    "CO-surface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "CO",
        UNIVERSITY: "Colorado State University",
        MANAGING_AGENCY: "State Land Board",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "18 Stat. 474-476",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "CO",
        LAYER: "SLB_Surface_Ownership",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["Beneficiar"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {"Penitentiary": "Penitentiary"},
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Acreage": ACRES,
            "County": COUNTY,
            "Township": TOWNSHIP,
            "Range": RANGE,
            "Section": SECTION,
            "Meridian": MERIDIAN,
        },
    },
    "CO-subsurface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "CO",
        UNIVERSITY: "Colorado State University",
        MANAGING_AGENCY: "State Land Board",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "18 Stat. 474-476",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "CO",
        LAYER: "SLB_Mineral_Estate",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["Beneficiar"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {"Penitentiary": "Penitentiary"},
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Acreage": ACRES,
            "County": COUNTY,
            "Township": TOWNSHIP,
            "Range": RANGE,
            "Section": SECTION,
            "Meridian": MERIDIAN,
            "Asset_Laye": ACTIVITY,
        },
    },
    "ID-surface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "ID",
        UNIVERSITY: "University of Idaho",
        MANAGING_AGENCY: "Department of Lands",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "26 Stat. 215-219 (1890)",
        DATA_SOURCE: "https://gis1.idl.idaho.gov/arcgis/rest/services/State_Ownership/MapServer/0",
        NET_ACRES: STATE_TRUST_DATA_SOURCE_DIRECTORY
        + "Net_Acreage_Percent_Ownership_Idaho.csv",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["SURF_ENDOWMENT"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            907: "907: 27.00% Charitable Institute, 65.00% Public School (Indemnity, Schools, Common Schools), 8.00% University of Idaho",
            5: "5: 100.00% Penitentiary",
            3: "3: 100.00% Charitable Institute",
            13: "13: 100.00% Dept. of Correction",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "GISACRES": ACRES,
        },
    },
    "ID-subsurface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "ID",
        UNIVERSITY: "University of Idaho",
        MANAGING_AGENCY: "Department of Lands",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "26 Stat. 215-219 (1890)",
        DATA_SOURCE: "https://gis1.idl.idaho.gov/arcgis/rest/services/State_Ownership/MapServer/1",
        NET_ACRES: STATE_TRUST_DATA_SOURCE_DIRECTORY
        + "Net_Acreage_Percent_Ownership_Idaho.csv",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["SUB_ENDOWMENT"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            916: "916: Dept. of Fish and Game, Penitentiary",
            932: "932: Dept. of Parks and Recreation, Penitentiary",
            3: "3: 100.00% Charitable Institute",
            13: "13: 100.00% Dept. of Correction",
            930: "930: Charitable Institute, Public School (Indemnity, Schools, Common Schools)",
            907: "907: 27.00% Charitable Institute, 65.00% Public School (Indemnity, Schools, Common Schools), 8.00% University of Idaho",
            5: "5: 100.00% Penitentiary",
            925: "925: Penitentiary, Public School (Indemnity, Schools, Common Schools)",
            912: "912: Charitable Institute, Dept. of Fish and Game",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "GISACRES": ACRES,
        },
    },
    "MT-surface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "MT",
        UNIVERSITY: "Montana State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        DATA_SOURCE: "https://gis.dnrc.mt.gov/arcgis/rest/services/DNRALL/BasemapService/MapServer/31",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["GrantID"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'PH'": "Pine Hills (State Industrial School)",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Acres": ACRES,
        },
    },
    "MT-subsurface-coal": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "MT",
        UNIVERSITY: "Montana State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "MT-coal",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["GrantID"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "PH": "Pine Hills (State Industrial School)",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "CoalGAcres": ACRES,
            "County1": COUNTY,
            "CoalGLegal": ALIQUOT,
            "Activity": ACTIVITY,
        },
    },
    "MT-subsurface-oil-and-gas": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "MT",
        UNIVERSITY: "Montana State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "MT-oil-and-gas",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["GrantID"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "PH": "Pine Hills (State Industrial School)",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "OG_GAcres": ACRES,
            "County1": COUNTY,
            "OG_GLegal": ALIQUOT,
            "Activity": ACTIVITY,
        },
    },
    "MT-subsurface-other-minerals": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "MT",
        UNIVERSITY: "Montana State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "MT-other-minerals",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["GrantID"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "PH": "Pine Hills (State Industrial School)",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Othr_GAcres": ACRES,
            "County1": COUNTY,
            "Othr_GLegal": ALIQUOT,
            "Activity": ACTIVITY,
        },
    },
    "NM-surface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "NM",
        UNIVERSITY: "New Mexico State University",
        MANAGING_AGENCY: "State Land Office",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "36 Stat. 557-579 , esp. 572-573 (1910)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "NM",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["Benef_Surf"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "14": "14: New Mexico Penitentiary",
            "17": "17: Charitable, Penal, and Reform",
            "29": "29: Charitable, Penal, and Reform",
            "38": "38: New Mexico Penitentiary",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Acres_Surf": ACRES,
            "Township": TOWNSHIP,
            "Range": RANGE,
            "Section": SECTION,
            "Meridian": MERIDIAN,
            "Aliquot": ALIQUOT,
        },
    },
    "NM-subsurface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "NM",
        UNIVERSITY: "New Mexico State University",
        MANAGING_AGENCY: "State Land Office",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "36 Stat. 557-579 , esp. 572-573 (1910)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "NM",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["Benef_SubS"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "14": "14: New Mexico Penitentiary",
            "17": "17: Charitable, Penal, and Reform",
            "29": "29: Charitable, Penal, and Reform",
            "38": "38: New Mexico Penitentiary",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "Acres_SubS": ACRES,
            "Township": TOWNSHIP,
            "Range": RANGE,
            "Section": SECTION,
            "Meridian": MERIDIAN,
            "Aliquot": ALIQUOT,
        },
    },
    "ND-surface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "ND",
        UNIVERSITY: "North Dakota State University",
        MANAGING_AGENCY: "Commissioner of University and School Lands",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "ND-surface.zip",
        ACTIVITY: "Recreation",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["Trust_Desc"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "ND INDUSTRIAL SCHOOL": "North Dakota Industrial School",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "ACRES": ACRES,
            "COUNTY": COUNTY,
            "TOWNSHIP": TOWNSHIP,
            "RANGE": RANGE,
            "SECTION": SECTION,
            "SUBDIVISIO": ALIQUOT,
        },
    },
    "ND-subsurface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "ND",
        UNIVERSITY: "North Dakota State University",
        MANAGING_AGENCY: "Commissioner of University and School Lands",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        DATA_SOURCE: "https://ndgishub.nd.gov/arcgis/rest/services/All_GovtLands_State/MapServer/2",
        ACTIVITY: "Oil and Gas Lease",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["TRUST"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {"'I'": "North Dakota Industrial School"},
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "GROSS_ACRES": ACRES,
            "COUNTY": COUNTY,
            "TWP": TOWNSHIP,
            "RNG": RANGE,
            "SEC": SECTION,
            "SUBDIVISION": ALIQUOT,
            "NET_ACRES": NET_ACRES,
        },
    },
    'SD-subsurface': {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: 'SD',
        UNIVERSITY: 'South Dakota State University',
        MANAGING_AGENCY: 'Commissioner of School and Public Lands',
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: '25 Stat. 676-684, esp. 679-81 (1889)',
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + 'SD-subsurface',
        ATTRIBUTE_LABEL_TO_FILTER_BY: ['*'],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "*": 'Training School'
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            'RECRDAREAN': ACRES,
            'QQSEC': ALIQUOT,
        },
    },
    'SD-surface': {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: 'SD',
        UNIVERSITY: 'South Dakota State University',
        MANAGING_AGENCY: 'Commissioner of School and Public Lands',
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: '25 Stat. 676-684, esp. 679-81 (1889)',
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + 'SD-surface',
        ATTRIBUTE_LABEL_TO_FILTER_BY: ['Land Class'],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            'DOC, JUVENILE PROGRAMS': 'Training School'
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            'RECRDAREAN': ACRES,
            'QQSEC': ALIQUOT,
        },
    },
    "UT-subsurface-oil-and-gas": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "UT",
        UNIVERSITY: "Utah State University",
        MANAGING_AGENCY: "Trust Lands Administration",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "28 Stat. 107-110 (1894)",
        DATA_SOURCE: "https://gis.trustlands.utah.gov/mapping/rest/services/Ownership_Oil_Gas/FeatureServer/0",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["bene_abrev"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'SYDC'": "Juvenile Justice Services",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "acres": ACRES,
            "township": TOWNSHIP,
            "range": RANGE,
            "county_name": COUNTY,
            "section_": SECTION,
            "legal_descr": ALIQUOT,
        },
    },
    "UT-subsurface-coal": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "UT",
        UNIVERSITY: "Utah State University",
        MANAGING_AGENCY: "Trust Lands Administration",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "28 Stat. 107-110 (1894)",
        DATA_SOURCE: "https://gis.trustlands.utah.gov/mapping/rest/services/Ownership_Coal/FeatureServer/0",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["bene_abrev"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'SYDC'": "Juvenile Justice Services",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "acres": ACRES,
            "township": TOWNSHIP,
            "range": RANGE,
            "county_name": COUNTY,
            "section_": SECTION,
            "legal_descr": ALIQUOT,
        },
    },
    "UT-subsurface-other-minerals": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "UT",
        UNIVERSITY: "Utah State University",
        MANAGING_AGENCY: "Trust Lands Administration",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "28 Stat. 107-110 (1894)",
        DATA_SOURCE: "https://gis.trustlands.utah.gov/mapping/rest/services/Ownership_Other_Mineral/FeatureServer/0",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["bene_abrev"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "'SYDC'": "Juvenile Justice Services",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "acres": ACRES,
            "township": TOWNSHIP,
            "range": RANGE,
            "county_name": COUNTY,
            "section_": SECTION,
            "legal_descr": ALIQUOT,
        },
    },
    "WA-timber": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "WA",
        UNIVERSITY: "Washington State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: TIMBER_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        DATA_SOURCE: "https://gis.dnr.wa.gov/site3/rest/services/Public_Boundaries/WADNR_PUBLIC_Cadastre_OpenData/MapServer/6/",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["TIMBER_TRUST_CD"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            6: "Charitable/Educational/Penal Reformatory Instit.",
            30: "Washington Dept. of Corrections",
        },
    },
    "WA-surface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "WA",
        UNIVERSITY: "Washington State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        DATA_SOURCE: "https://gis.dnr.wa.gov/site3/rest/services/Public_Boundaries/WADNR_PUBLIC_Cadastre_OpenData/MapServer/6/",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["SURFACE_TRUST_CD"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            6: "Charitable/Educational/Penal Reformatory Instit.",
            30: "Washington Dept. of Corrections",
        },
    },
    "WA-subsurface": {
        DOWNLOAD_TYPE: API_QUERY_DOWNLOAD_TYPE,
        STATE: "WA",
        UNIVERSITY: "Washington State University",
        MANAGING_AGENCY: "Department of Natural Resources",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "25 Stat. 676-684, esp. 679-81 (1889)",
        DATA_SOURCE: "https://gis.dnr.wa.gov/site3/rest/services/Public_Boundaries/WADNR_PUBLIC_Cadastre_OpenData/MapServer/6/",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["MINERAL_TRUST_CD"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            6: "Charitable/Educational/Penal Reformatory Instit.",
            30: "Washington Dept. of Corrections",
        },
    },
    "WY-subsurface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "WY",
        UNIVERSITY: "University of Wyoming",
        MANAGING_AGENCY: "Office of State Lands and Investment",
        RIGHTS_TYPE: SUBSURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "26 Stat. 222-226 (1890)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "WY-subsurface",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["FundCode"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "CR": "Board of Charities and Reform",
            "PR": "Penal, Reform or Educational Institutions",
            "PE": "Penitentiary",
            "SR": "State Charitable, Educational, Penal & Reform Inst",
            "CD": "Department of Corrections",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "SubsurfaceAcres": ACRES,
            "Township_Temp": TOWNSHIP,
            "Range_Temp": RANGE,
            "FirstDivision_Temp": SECTION,
        },
    },
    "WY-surface": {
        DOWNLOAD_TYPE: SHAPEFILE_DOWNLOAD_TYPE,
        STATE: "WY",
        UNIVERSITY: "University of Wyoming",
        MANAGING_AGENCY: "Office of State Lands and Investment",
        RIGHTS_TYPE: SURFACE_RIGHTS_TYPE,
        STATE_ENABLING_ACT: "26 Stat. 222-226 (1890)",
        LOCAL_DATA_SOURCE: STATE_TRUST_DATA_SOURCE_DIRECTORY + "WY-surface",
        ATTRIBUTE_LABEL_TO_FILTER_BY: ["FundCode"],
        ATTRIBUTE_CODE_TO_ALIAS_MAP: {
            "CR": "Board of Charities and Reform",
            "PR": "Penal, Reform or Educational Institutions",
            "PE": "Penitentiary",
            "SR": "State Charitable, Educational, Penal & Reform Inst",
            "CD": "Department of Corrections",
        },
        EXISTING_COLUMN_TO_FINAL_COLUMN_MAP: {
            "SurfaceAcres": ACRES,
            "Township_Temp": TOWNSHIP,
            "Range_Temp": RANGE,
            "FirstDivision_Temp": SECTION,
        },
    },
}
