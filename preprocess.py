import os
import re
import pandas as pd
from joblib import dump

RAW_CSV_FILES = [
    'ausgrid_data/Solar home half-hour data - 1 July 2010 to 30 June 2011 (2)/2010-2011 Solar home electricity data.csv',
    'ausgrid_data/Solar home half-hour data - 1 July 2011 to 30 June 2012 (3)/2011-2012 Solar home electricity data v2.csv',
    'ausgrid_data/Solar home half-hour data - 1 July 2012 to 30 June 2013 (2)/2012-2013 Solar home electricity data v2.csv'
]

OUTPUT_DIR = './Ausgrid_processed_for_diffusion'
NUM_HOUSES = 300
TIMEZONE = 'Australia/Sydney'

def clean_ausgrid_data():
    """
    Loads, cleans, and reshapes the raw Ausgrid data into a useful format.
    """
    print("Starting data cleanup for Ausgrid dataset...")

    # Load and combine all the raw CSV files.
    all_data = []
    for file_path in RAW_CSV_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: Can't find '{file_path}'. Skipping it.")
            continue
        try:
            df = pd.read_csv(file_path, low_memory=False)
            all_data.append(df)
            print(f"Loaded '{os.path.basename(file_path)}'.")
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")

    if not all_data:
        raise FileNotFoundError("No data files were loaded. Check your paths in RAW_CSV_FILES.")

    # Stick all the files together into one big table.
    combined_df = pd.concat(all_data, ignore_index=True)

    # Make sure customer IDs are clean numbers.
    combined_df.dropna(subset=['Customer'], inplace=True)
    combined_df['Customer'] = combined_df['Customer'].astype(int)

    #  Pick the first NUM_HOUSES and drop columns we don't need.
    all_customers = sorted(combined_df['Customer'].unique())
    selected_customers = all_customers[:NUM_HOUSES]
    df = combined_df[combined_df['Customer'].isin(selected_customers)]
    df = df.drop(columns=['Generator Capacity', 'Postcode'], errors='ignore')
    print(f"Selected {len(selected_customers)} houses to process.")

    #  Reshape the data. The original data is "wide" with a column for each
    time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
    time_cols = [c for c in df.columns if time_pattern.match(c)]
    id_vars = [c for c in df.columns if c not in time_cols]
    long_df = df.melt(id_vars=id_vars, value_vars=time_cols, var_name='time', value_name='value')
    
    # 4. Create proper timestamps.
    long_df['local_time'] = pd.to_datetime(long_df['date'] + ' ' + long_df['time'])
    long_df['local_time'] = long_df['local_time'].dt.tz_localize(
        TIMEZONE, nonexistent='shift_forward', ambiguous='NaT'
    ).dt.tz_localize(None)
    long_df = long_df.dropna(subset=['local_time']) # Drop rows where timestamp creation failed

    # 5. Turn consumption categories ('GC', 'GG', 'CL') into separate columns.
    pivoted_df = long_df.pivot_table(
        index=['Customer', 'local_time'],
        columns='Consumption Category',
        values='value'
    ).reset_index()
    pivoted_df.columns.name = None
    pivoted_df = pivoted_df.fillna(0)
    
    # 6. Calculate 'grid_usage' and 'solar_generation'.
    #    grid = (General Consumption + Controlled Load) - Gross Generation
    pivoted_df['grid_usage'] = pivoted_df['GC'] + pivoted_df['CL'] - pivoted_df['GG']
    pivoted_df['solar_generation'] = pivoted_df['GG'].clip(lower=0)
    
    # 7. Final check to make sure we only have full days of data.
    pivoted_df['date_only'] = pivoted_df['local_time'].dt.date
    daily_counts = pivoted_df.groupby(['Customer', 'date_only']).size()
    complete_days = daily_counts[daily_counts == 48].reset_index()[['Customer', 'date_only']]
    
    final_df = pd.merge(pivoted_df, complete_days, on=['Customer', 'date_only'])
    final_df = final_df.drop(columns=['date_only', 'GC', 'CL', 'GG'])
    print(f"Cleaned data has {final_df.shape[0]} total readings.")
    
    return final_df

def save_data_for_model(clean_df):
    """
    Saves the cleaned data into individual files for each house.
    """
    print("\nSaving cleaned data into individual house files...")
    per_house_dir = os.path.join(OUTPUT_DIR, 'per_house')
    os.makedirs(per_house_dir, exist_ok=True)
    
    house_ids = sorted(clean_df['Customer'].unique())

    for house_id in house_ids:
        house_df = clean_df[clean_df['Customer'] == house_id][['local_time', 'grid_usage', 'solar_generation']].copy()
        house_df.rename(columns={'local_time': 'timestamp'}, inplace=True)
        
        output_path = os.path.join(per_house_dir, f'house_{house_id}.csv')
        house_df.to_csv(output_path, index=False)
        
    print(f"Saved {len(house_ids)} individual house files to '{per_house_dir}/'")

if __name__ == "__main__":
    cleaned_data = clean_ausgrid_data()
    
    if cleaned_data is not None:
        save_data_for_model(cleaned_data)
        print("\nAll done!")
