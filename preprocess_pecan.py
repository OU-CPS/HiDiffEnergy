import os
import pandas as pd
import numpy as np
from typing import Dict

class EnergyConfig:
    FINAL_OUTPUT_DIR = './pecan_processed_for_diffusion'
    TIME_COLUMN = 'local_15min'
    ID_COLUMN = 'dataid'
    GRID_COLUMN = 'grid'
    SOLAR_COLUMN = 'solar'
    SOLAR_THRESHOLD = 0.005
    COLUMNS_TO_KEEP = ['dataid', 'local_15min', 'grid', 'solar']
    COLUMNS_TO_COMBINE = [('solar', 'solar2', 'solar')]

def process_location_data(raw_csv_path: str, common_output_dir: str, 
                          location: str, timezone: str, 
                          config=EnergyConfig) -> Dict:
    print(f"\nProcessing data for location: {location.upper()}...")
    print("-" * 60)
    
    # Load
    print(f"Loading raw data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    print(f"  Initial shape: {df.shape}")
    
    # Timestamps
    print("Processing timestamps...")
    df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], utc=True)
    df[config.TIME_COLUMN] = df[config.TIME_COLUMN].dt.tz_convert(timezone)
    df[config.TIME_COLUMN] = df[config.TIME_COLUMN].dt.tz_localize(None)
    
    # Clean
    print("Cleaning and preparing data...")
    df = df.fillna(0)
    
    # Combine
    if hasattr(config, 'COLUMNS_TO_COMBINE'):
        for col1, col2, new_col in config.COLUMNS_TO_COMBINE:
            if col1 in df.columns and col2 in df.columns:
                df[new_col] = df[col1] + df[col2]
                df = df.drop(columns=[col2])
                print(f"  Combined '{col1}' and '{col2}' into '{new_col}'")
    
    df = df[config.COLUMNS_TO_KEEP]
    
    # Clamp solar
    if config.SOLAR_COLUMN in df.columns:
        df[config.SOLAR_COLUMN] = df[config.SOLAR_COLUMN].clip(lower=0)
        df[config.SOLAR_COLUMN] = df[config.SOLAR_COLUMN].where(
            df[config.SOLAR_COLUMN] >= config.SOLAR_THRESHOLD, 0)
        print("  Clamped low solar values to zero.")

    house_ids = sorted(df[config.ID_COLUMN].unique())
    print(f"  Found {len(house_ids)} unique house IDs.")
    
    # Pivot
    print("Pivoting data to a wide format...")
    pivoted = df.pivot_table(
        index=config.TIME_COLUMN,
        columns=config.ID_COLUMN,
        values=[config.GRID_COLUMN, config.SOLAR_COLUMN],
        aggfunc='first'
    )
    
    # Flatten columns
    pivoted.columns = [f"{metric}_{house_id}" for metric, house_id in pivoted.columns]
    
    # Drop NA
    initial_rows = len(pivoted)
    pivoted = pivoted.dropna().reset_index()
    print(f"  Removed {initial_rows - len(pivoted)} rows with missing values.")
    
    # Ensure days
    print("Filtering for complete days (96 intervals)...")
    pivoted['date'] = pivoted[config.TIME_COLUMN].dt.date
    daily_counts = pivoted.groupby('date').size()
    complete_days = daily_counts[daily_counts == 96].index
    
    pivoted = pivoted[pivoted['date'].isin(complete_days)].drop(columns=['date'])
    print(f"  Retained data for {len(complete_days)} complete days.")

    # Save files
    house_dir = os.path.join(common_output_dir, 'per_house')
    os.makedirs(house_dir, exist_ok=True)
    
    files_created = []
    for house_id in house_ids:
        house_df = pivoted[[config.TIME_COLUMN, f'grid_{house_id}', f'solar_{house_id}']].copy()
        
        # Rename
        house_df = house_df.rename(columns={
            config.TIME_COLUMN: 'timestamp',
            f'grid_{house_id}': 'grid_usage',
            f'solar_{house_id}': 'solar_generation'
        })
        
        # Write CSV
        house_filename = f'{location}_house_{house_id}.csv'
        house_path = os.path.join(house_dir, house_filename)
        house_df.to_csv(house_path, index=False)
        files_created.append(house_path)
    
    print(f"Saved {len(house_ids)} individual house files to '{house_dir}'")
    print(f"Processing complete for {location.upper()}.")

    return {
        'location': location,
        'final_shape': pivoted.shape,
        'date_range': (pivoted[config.TIME_COLUMN].min(), pivoted[config.TIME_COLUMN].max()),
        'num_houses': len(house_ids),
        'files_created': len(files_created)
    }

if __name__ == "__main__":
    # Inputs
    locations_to_process = {
        'austin': {
            'raw_path': './raw_data/15minute_data_austin.csv',
            'timezone': 'America/Chicago'
        },
        'newyork': {
            'raw_path': './raw_data/15minute_data_newyork.csv', 
            'timezone': 'America/New_York'
        }
    }
    
    final_output_dir = EnergyConfig.FINAL_OUTPUT_DIR
    
    # Run
    all_results = []
    for loc, info in locations_to_process.items():
        if os.path.exists(info['raw_path']):
            result = process_location_data(
                raw_csv_path=info['raw_path'],
                common_output_dir=final_output_dir,
                location=loc,
                timezone=info['timezone']
            )
            all_results.append(result)
        else:
            print(f"WARNING: Raw data file not found. Skipping {loc.upper()}.")
            print(f"  Expected path: {info['raw_path']}")
            
    # Summary
    total_files = sum(r.get('files_created', 0) for r in all_results)
    print(f"Total individual house files created: {total_files}")
    print(f"All files saved in: {os.path.join(final_output_dir, 'per_house')}")
