import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MultiHouseDataset(torch.utils.data.Dataset):
    """Optimized PyTorch Dataset for time series data from multiple houses."""
    # MODIFIED: Added step_size to create tumbling (non-overlapping) windows if desired.
    def __init__(self, data_dir: str, window_size: int = 96, step_size: int = 1,
                 scaler_path: str = 'global_scaler.gz', cache_in_memory: bool = True,
                 dtype: torch.dtype = torch.float32, limit_to_one_year: bool = True):
        """
        Args:
            data_dir: Path to directory with house CSV files.
            window_size: The sequence length for each sample.
            step_size: The step between the start of consecutive windows.
                       (Set to 1 for a sliding window, set to window_size for a non-overlapping window).
            scaler_path: Path to save or load the global scaler.
            cache_in_memory: Whether to cache all normalized data in memory for faster access.
            dtype: Data type for tensors (float32 vs float16 for memory savings).
            limit_to_one_year: If True, only use the first year of data for each house.
        """
        self.window_size = window_size
        self.step_size = step_size  # ADDED: Store step_size
        self.cache_in_memory = cache_in_memory
        self.dtype = dtype
        self.limit_to_one_year = limit_to_one_year
        
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        print(f"Found {len(all_files)} house files in '{data_dir}'.")
        
        self.num_houses = len(all_files)
        
        # 1. Load all data from disk more efficiently
        print("Reading all house data from disk...")
        if self.limit_to_one_year:
            print("INFO: Limiting data to the first year (17,520 samples) for each house.")
            
        data_per_house = []
        timestamps_per_house = []
    
        SAMPLES_PER_YEAR = 17520 # (48 samples/day * 365 days)
        
        for filename in all_files:
            df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=['timestamp'])
            timestamps_per_house.append(df['timestamp'].values)
            time_series_values = df[['grid_usage', 'solar_generation']].values.astype(np.float32)

            # Slice the data to one year if the flag is set
            if self.limit_to_one_year:
                time_series_values = time_series_values[:SAMPLES_PER_YEAR]

            # --- NEW: Generate time-of-day features ---
            num_timesteps = len(time_series_values)
            # Create a time index from 0 to 47 for each day
            timesteps_of_day = np.arange(num_timesteps) % 48 

            # Create sine and cosine features
            sin_time = np.sin(2 * np.pi * timesteps_of_day / 48.0).astype(np.float32)
            cos_time = np.cos(2 * np.pi * timesteps_of_day / 48.0).astype(np.float32)

            # Combine with original data
            time_series_values = np.concatenate([
                time_series_values, 
                sin_time[:, np.newaxis], # Reshape to (n, 1)
                cos_time[:, np.newaxis]  # Reshape to (n, 1)
            ], axis=1)
            # --- END OF NEW CODE ---

            data_per_house.append(time_series_values)

        # 2. Fit a global scaler and normalize data
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print("Fitting a single global scaler on all houses...")
            combined_data = np.vstack(data_per_house)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(combined_data)
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        
        # Store normalized data
        if self.cache_in_memory:
            print("Caching normalized data in memory as tensors...")
            self.normalized_data_per_house = []
            for series in data_per_house:
                normalized = scaler.transform(series)
                tensor_data = torch.from_numpy(normalized).to(dtype=self.dtype)
                self.normalized_data_per_house.append(tensor_data)
        else:
            self.normalized_data_per_house = []
            for series in data_per_house:
                self.normalized_data_per_house.append(scaler.transform(series))
        
        del data_per_house
            
        # 3. Pre-compute window mappings for O(1) lookup
        # --- MODIFIED SECTION: This logic now uses step_size ---
        print("Pre-computing window mappings...")
        
        # Calculate windows per house based on the step size
        self.windows_per_house = [(len(d) - self.window_size) // self.step_size + 1 for d in self.normalized_data_per_house]
        self.cumulative_windows = np.cumsum([0] + self.windows_per_house)
        self.total_windows = self.cumulative_windows[-1]

        # Pre-compute house indices and local start positions
        self.sample_to_house = np.empty(self.total_windows, dtype=np.int32)
        self.sample_to_local_idx = np.empty(self.total_windows, dtype=np.int32)
        self.sample_to_day_of_week = np.empty(self.total_windows, dtype=np.int32)
        self.sample_to_day_of_year = np.empty(self.total_windows, dtype=np.int32)

        
        for house_idx in range(self.num_houses):
            start_global_idx = self.cumulative_windows[house_idx]
            end_global_idx = self.cumulative_windows[house_idx + 1]
            num_windows_for_this_house = self.windows_per_house[house_idx]

            self.sample_to_house[start_global_idx:end_global_idx] = house_idx
            
            # The key change: local index now jumps by step_size instead of 1
            local_indices = np.arange(num_windows_for_this_house) * self.step_size
            self.sample_to_local_idx[start_global_idx:end_global_idx] = local_indices

            house_timestamps = pd.Series(timestamps_per_house[house_idx][local_indices])
            self.sample_to_day_of_week[start_global_idx:end_global_idx] = house_timestamps.dt.dayofweek
            self.sample_to_day_of_year[start_global_idx:end_global_idx] = house_timestamps.dt.dayofyear - 1 # (0-365)
        # --- END OF MODIFIED SECTION ---

        print(f"Dataset initialized. Total windows: {self.total_windows} from {self.num_houses} houses.")
        memory_usage = sum(data.numel() * data.element_size() for data in self.normalized_data_per_house) / 1e6 if self.cache_in_memory else 0
        print(f"Memory usage for cached tensors: {memory_usage:.1f} MB")

    def __len__(self) -> int:
        return self.total_windows

    # --- MODIFICATION: Change the return type ---
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError("Index out of range")

        house_index = self.sample_to_house[idx]
        local_start_pos = self.sample_to_local_idx[idx]
        
        window_data = self.normalized_data_per_house[house_index][local_start_pos : local_start_pos + self.window_size]
        
        # --- NEW: Create a dictionary of all conditions ---
        conditions = {
            "house_id": torch.tensor(house_index, dtype=torch.long),
            "day_of_week": torch.tensor(self.sample_to_day_of_week[idx], dtype=torch.long),
            "day_of_year": torch.tensor(self.sample_to_day_of_year[idx], dtype=torch.long),
        }
        
        return window_data, conditions


    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        if self.cache_in_memory:
            tensor_memory = sum(data.numel() * data.element_size() for data in self.normalized_data_per_house) / 1e6
        else:
            tensor_memory = 0
        
        mapping_memory = (self.sample_to_house.nbytes + self.sample_to_local_idx.nbytes) / 1e6
        
        return {
            'tensor_cache_mb': tensor_memory,
            'mapping_arrays_mb': mapping_memory,
            'total_mb': tensor_memory + mapping_memory
        }

# Additional utility class for even faster latent dataset creation
class LatentDataset(torch.utils.data.Dataset):
    """Optimized dataset for pre-computed latent vectors."""
    def __init__(self, latent_vectors: torch.Tensor, house_ids: torch.Tensor):
        assert len(latent_vectors) == len(house_ids), "Latent vectors and house IDs must have same length"
        self.latent_vectors = latent_vectors.contiguous()
        self.house_ids = house_ids.contiguous()
        
    def __len__(self) -> int:
        return len(self.latent_vectors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latent_vectors[idx], self.house_ids[idx]

# Example Usage and Benchmarking
if __name__ == "__main__":
    import time
    
    DATA_DIRECTORY = './Ausgrid_processed_for_diffusion/per_house/'
    
    if os.path.exists(DATA_DIRECTORY):
        print("=== Testing Non-Overlapping Window DataLoader ===")
        
        start_time = time.time()
        # To get non-overlapping 2-day windows, set step_size = window_size
        dataset = MultiHouseDataset(data_dir=DATA_DIRECTORY, window_size=96, step_size=96)
        init_time = time.time() - start_time
        print(f"Dataset initialization: {init_time:.2f}s")
        print(f"Memory usage: {dataset.get_memory_usage()}")
        
        if len(dataset) > 0:
            first_sample, first_house_id = dataset[0]
            print(f"\nSample data shape: {first_sample.shape}")
            print(f"Sample house ID: {first_house_id.item()}")
            print(f"Data type: {first_sample.dtype}")
            print(f"Total unique houses: {dataset.num_houses}")
            
            second_sample, _ = dataset[1]
            # The second sample from the same house should be exactly 96 steps after the first
            print(f"The second sample starts {dataset.sample_to_local_idx[1]} steps after the first.")
            
    else:
        print(f"ERROR: Data directory not found at '{DATA_DIRECTORY}'")