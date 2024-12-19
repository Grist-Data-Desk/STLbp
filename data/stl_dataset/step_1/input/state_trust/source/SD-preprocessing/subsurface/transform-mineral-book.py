import pandas as pd
import argparse
from typing import List, Dict, Set, Tuple
import re

def load_plss_inference(filename: str) -> pd.DataFrame:
    """
    Loads the PLSS inference data for lookups.
    Returns DataFrame indexed by (County, TWNSHPNO, RANGENO).
    """
    plss_df = pd.read_csv(filename)
    # Clean up county names to match our format
    plss_df['County'] = plss_df['County'].str.replace(' County', '').str.strip()
    # Create and sort multi-index
    plss_df = plss_df.set_index(['County', 'TWNSHPNO', 'RANGENO'])
    return plss_df.sort_index()

def normalize_township_range(value: str, is_township: bool = True) -> Tuple[str, int]:
    """
    Normalizes a township or range value with correct padding.
    Returns tuple of (normalized value, numeric value).
    
    Examples:
    8 -> "0080"
    10 -> "0100"
    11 -> "0110"
    100 -> "1000"
    """
    value = str(value).strip().upper()
    
    # First check if direction is explicitly provided
    match = re.match(r'^(\d+)([NSEW])$', value)
    if match:
        num, direction = match.groups()
        num_val = int(num)
        # Proper padding
        num_str = str(num_val)
        if len(num_str) == 1:
            num_padded = f"00{num_str}0"
        elif len(num_str) == 2:
            num_padded = f"0{num_str}0"
        else:
            num_padded = f"{num_str}0"
        return f"{num_padded}{direction}", num_val
    
    try:
        num_val = int(float(value))
        # Proper padding
        num_str = str(num_val)
        if len(num_str) == 1:
            num_padded = f"00{num_str}0"
        elif len(num_str) == 2:
            num_padded = f"0{num_str}0"
        else:
            num_padded = f"{num_str}0"
        return f"{num_padded}0", num_val
    except ValueError as e:
        raise ValueError(f"Invalid format for {'township' if is_township else 'range'}: {value}") from e

def get_plss_info(county: str, township_num: int, range_num: int, plss_df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Looks up meridian code and directions from PLSS inference data.
    Returns tuple of (meridian_code, township_dir, range_dir).
    """
    # Clean county name for lookup
    county = county.replace(" County", "").strip()
    try:
        # Get the row for this county, township, range combination
        row = plss_df.loc[(county, township_num, range_num)]
        # If multiple matches exist, take the first one
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
            
        # Get the numeric meridian code
        meridian_code = str(int(row['PRINMERCD'])).zfill(2)
        
        return (
            meridian_code,
            row['TWNSHPDIR'],
            row['RANGEDIR']
        )
    except KeyError:
        raise KeyError(f"No PLSS info found for {county}, T{township_num}, R{range_num}")

def validate_quarter_quarter(code: str) -> bool:
    """
    Validates that a quarter-quarter code follows the correct NESW pattern.
    Must be exactly 4 characters, each position can only be N/S then E/W.
    """
    if len(code) != 4:
        return False
    return all(code[i] in valid for i, valid in enumerate(['NS', 'EW', 'NS', 'EW']))

def get_quarter_quarters(base_quarter: str = None) -> Set[str]:
    """
    Returns all quarter-quarters for a given quarter section or entire section if base_quarter is None.
    For a quarter section (e.g., 'NE'), generates all possible QQs where NE is the quarter position.
    
    Examples:
        get_quarter_quarters()     -> {"NENE", "NENW", "NWNE", "NWNW", ...} 
        get_quarter_quarters("NE") -> {"NENE", "NWNE", "SENE", "SWNE"}  # NE is quarter position
    """
    quarters = ['NE', 'NW', 'SE', 'SW']
    if base_quarter:
        # For a specific quarter, generate all possible QQs where it's the quarter position
        return {f"{qq}{base_quarter}" for qq in quarters}
    else:
        # For a whole section, generate all possible combinations
        return {f"{qq1}{qq2}" for qq1 in quarters for qq2 in quarters}

def get_half_section_quarters(half_type: str) -> Set[str]:
    """
    Returns all quarter-quarters for a half section (N2, S2, E2, W2).
    
    Examples:
        'N2' -> All QQs that contain N in second or fourth position
        'S2' -> All QQs that contain S in second or fourth position
        'E2' -> All QQs that contain E in second or fourth position
        'W2' -> All QQs that contain W in second or fourth position
    """
    quarters = ['NE', 'NW', 'SE', 'SW']
    if half_type == 'N2':
        # Generate all QQs that have N in either the quarter-quarter or quarter position
        return {f"{qq1}{qq2}" for qq1 in quarters for qq2 in quarters if 'N' in qq2}
    elif half_type == 'S2':
        # Generate all QQs that have S in either the quarter-quarter or quarter position
        return {f"{qq1}{qq2}" for qq1 in quarters for qq2 in quarters if 'S' in qq2}
    elif half_type == 'E2':
        # Generate all QQs that have E in either the quarter-quarter or quarter position
        return {f"{qq1}{qq2}" for qq1 in quarters for qq2 in quarters if 'E' in qq2}
    elif half_type == 'W2':
        # Generate all QQs that have W in either the quarter-quarter or quarter position
        return {f"{qq1}{qq2}" for qq1 in quarters for qq2 in quarters if 'W' in qq2}
    return set()

def expand_single_description(desc: str) -> Set[str]:
    """
    Expands a single part of a legal description into quarter-quarters.
    """
    desc = desc.strip().upper()
    
    # Handle basic patterns
    if desc == 'ALL':
        return get_quarter_quarters()
    elif desc in ['N2', 'S2', 'E2', 'W2']:
        return get_half_section_quarters(desc)
    elif desc.endswith('4') and desc[:-1] in ['NE', 'NW', 'SE', 'SW']:
        return get_quarter_quarters(desc[:-1])
    elif validate_quarter_quarter(desc):
        return {desc}
    
    # Handle composite descriptions (e.g., N2SE)
    match = re.match(r'([NS]2|[EW]2)([NSEW]{2})', desc)
    if match:
        half, quarter = match.groups()
        base_quarters = get_quarter_quarters(quarter)
        # Filter based on the half designation
        if half[0] == 'N':
            return {q for q in base_quarters if 'N' in q[2:]}
        elif half[0] == 'S':
            return {q for q in base_quarters if 'S' in q[2:]}
        elif half[0] == 'E':
            return {q for q in base_quarters if 'E' in q[2:]}
        elif half[0] == 'W':
            return {q for q in base_quarters if 'W' in q[2:]}
    
    return set()

def extract_lots(legal_desc: str) -> tuple[Set[str], str]:
    """
    Extracts lot numbers from a legal description and returns the lot numbers
    and the remaining description with lot information completely removed.
    """
    lots = set()
    # Find all lot numbers
    lot_matches = re.findall(r'LOT\s*(\d+)|LOTS\s*((?:\d+(?:\s*,\s*\d+)*))', legal_desc, re.IGNORECASE)
    
    # Process single lots and comma-separated lot lists
    for single, group in lot_matches:
        if single:
            lots.add(f"L{single}")
        if group:
            for num in re.findall(r'\d+', group):
                lots.add(f"L{num}")
    
    # Remove all lot references
    cleaned_desc = re.sub(r'LOTS?\s*\d+(?:\s*,\s*\d+)*,?\s*', '', legal_desc, flags=re.IGNORECASE)
    cleaned_desc = re.sub(r';\s*$', '', cleaned_desc)  # Remove trailing semicolon if present
    
    return lots, cleaned_desc.strip()

def parse_legal_description(legal_desc: str) -> List[str]:
    """
    Parses a legal description into individual parts.
    Returns list of 4-character quarter-quarters or 2-character lot numbers.
    """
    # Clean up the description
    legal_desc = legal_desc.upper().replace(" - T", "").strip()
    
    # Handle "ALL" case immediately
    if legal_desc == "ALL":
        return sorted(get_quarter_quarters())
    
    # Extract lots and get cleaned description
    lots, cleaned_desc = extract_lots(legal_desc)
    parts = set(lots)  # Start with the lots we found
    
    # Split remaining description into parts and process each
    if cleaned_desc:
        desc_parts = [p.strip() for p in cleaned_desc.split(',') if p.strip()]
        for part in desc_parts:
            expanded = expand_single_description(part)
            if expanded:
                parts.update(expanded)
    
    return sorted(parts)

def generate_ids(row: Dict, plss_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Generates all IDs for a given row"""
    # First normalize township and range, getting both the formatted string and numeric values
    township_str, township_num = normalize_township_range(str(row['Township']), is_township=True)
    range_str, range_num = normalize_township_range(str(row['Range']), is_township=False)
    
    # Look up PLSS info - meridian_code will be 2-digit string
    meridian_code, township_dir, range_dir = get_plss_info(row['County'], township_num, range_num, plss_df)
    
    # If township already includes direction (ends in N/S), keep it
    if township_str[-1] in 'NS':
        township_dir = township_str[-1]
        township_str = township_str[:-1]  # Remove any trailing placeholder
    else:
        township_str = township_str[:-1]  # Remove placeholder '0'
        
    # Remove placeholder '0' from range_str
    range_str = range_str[:-1]
    
    # Build properly formatted strings
    township_final = f"{township_str}{township_dir}"
    range_final = f"{range_str}{range_dir}"
    
    # Correctly pad section number
    section_num = int(row['Section'])
    section = f"{section_num:02d}0"
    
    # Build IDs using proper padding
    plssid = f"SD{meridian_code}{township_final}{range_final}0"
    frstdivid = f"{plssid}SN{section}"
    
    parts = parse_legal_description(row['Description and Class'])
    secdivids = []
    legal_desc_short = []
    
    for part in parts:
        if part.startswith('L'):
            secdivids.append(f"{frstdivid}{part}")
            legal_desc_short.append(part)
        else:
            # Add 'A' prefix to both SECDIVID and Legal Description Short
            secdivids.append(f"{frstdivid}A{part}")
            legal_desc_short.append(f"A{part}")
    
    return {
        'MERIDIAN': meridian_code,
        'Legal Description Short': legal_desc_short,
        'PLSSID': plssid,
        'FRSTDIVID': frstdivid,
        'SECDIVID': secdivids
    }

def main():
    parser = argparse.ArgumentParser(description='Transform mineral book CSV to include PLSS IDs')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('mapping_file', help='PLSS inference mapping CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    args = parser.parse_args()
    
    # Load PLSS inference data
    plss_df = load_plss_inference(args.mapping_file)
    
    # Load and process mineral book data
    df = pd.read_csv(args.input_file)
    
    results = []
    for _, row in df.iterrows():
        try:
            ids = generate_ids(row.to_dict(), plss_df)
            for secdivid, legal_desc_short in zip(ids['SECDIVID'], ids['Legal Description Short']):
                result = row.to_dict()
                result.update({
                    'MERIDIAN': ids['MERIDIAN'],
                    'Legal Description Short': legal_desc_short,
                    'PLSSID': ids['PLSSID'],
                    'FRSTDIVID': ids['FRSTDIVID'],
                    'SECDIVID': secdivid
                })
                results.append(result)
        except Exception as e:
            print(f"Error processing row: {row}\nError: {str(e)}")
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    print(f"Processed {len(df)} input rows into {len(output_df)} output rows")

if __name__ == "__main__":
    main()