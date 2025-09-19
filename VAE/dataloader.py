import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class MultiHouseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, window_size: int = 96, step_size: int = 1,
                 scaler_path: str = 'global_scaler.gz', cache_in_memory: bool = True,
                 dtype: torch.dtype = torch.float32, limit_to_one_year: bool = False):
        self.window_size = window_size
        self.step_size = step_size
        self.cache_in_memory = cache_in_memory
        self.dtype = dtype
        self.limit_to_one_year = limit_to_one_year
        
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        print(f"Found {len(all_files)} house files in '{data_dir}'.")
        
        self.num_houses = len(all_files)
        
        print("Reading all house data.")
        if self.limit_to_one_year:
            print("INFO: Limiting data to the first year (17,520 samples)")
            
        data_per_house = []
        SAMPLES_PER_YEAR = 17520
        
        for filename in all_files:
            df = pd.read_csv(os.path.join(data_dir, filename))
            time_series_values = df[['grid_usage', 'solar_generation']].values.astype(np.float32)

            if self.limit_to_one_year:
                time_series_values = time_series_values[:SAMPLES_PER_YEAR]

            # Generate time-of-day features using sine and cosine for cyclical encoding
            num_timesteps = len(time_series_values)
            timesteps_of_day = np.arange(num_timesteps) % 48 
            sin_time = np.sin(2 * np.pi * timesteps_of_day / 48.0).astype(np.float32)
            cos_time = np.cos(2 * np.pi * timesteps_of_day / 48.0).astype(np.float32)

            time_series_values = np.concatenate([
                time_series_values, 
                sin_time[:, np.newaxis],
                cos_time[:, np.newaxis]
            ], axis=1)

            data_per_house.append(time_series_values)

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            
            combined_data = np.vstack(data_per_house)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(combined_data)
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        
        if self.cache_in_memory:
           
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
            
        self.windows_per_house = [(len(d) - self.window_size) // self.step_size + 1 for d in self.normalized_data_per_house]
        self.cumulative_windows = np.cumsum([0] + self.windows_per_house)
        self.total_windows = self.cumulative_windows[-1]

        self.sample_to_house = np.empty(self.total_windows, dtype=np.int32)
        self.sample_to_local_idx = np.empty(self.total_windows, dtype=np.int32)
        
        for house_idx in range(self.num_houses):
            start_global_idx = self.cumulative_windows[house_idx]
            end_global_idx = self.cumulative_windows[house_idx + 1]
            num_windows_for_this_house = self.windows_per_house[house_idx]

            self.sample_to_house[start_global_idx:end_global_idx] = house_idx
            
            local_indices = np.arange(num_windows_for_this_house) * self.step_size
            self.sample_to_local_idx[start_global_idx:end_global_idx] = local_indices

        print(f"Dataset initialized. Total windows: {self.total_windows} from {self.num_houses} houses.")
        memory_usage = sum(data.numel() * data.element_size() for data in self.normalized_data_per_house) / 1e6 if self.cache_in_memory else 0

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError("Index out of range")

        house_index = self.sample_to_house[idx]
        local_start_pos = self.sample_to_local_idx[idx]
        
        if self.cache_in_memory:
            window_data = self.normalized_data_per_house[house_index][local_start_pos : local_start_pos + self.window_size]
        else:
            window_data = self.normalized_data_per_house[house_index][local_start_pos : local_start_pos + self.window_size]
            window_data = torch.from_numpy(window_data).to(dtype=self.dtype)
        
        house_id = torch.tensor(house_index, dtype=torch.long)
        
        return window_data, house_id

    def get_memory_usage(self) -> dict:
        tensor_memory = sum(d.numel() * d.element_size() for d in self.normalized_data_per_house) / 1e6 if self.cache_in_memory else 0
        mapping_memory = (self.sample_to_house.nbytes + self.sample_to_local_idx.nbytes) / 1e6
        
        return {
            'tensor_cache_mb': tensor_memory,
            'mapping_arrays_mb': mapping_memory,
            'total_mb': tensor_memory + mapping_memory
        }


class LatentDataset(torch.utils.data.Dataset):
    
    def __init__(self, latent_vectors: torch.Tensor, house_ids: torch.Tensor):
        assert len(latent_vectors) == len(house_ids), "Latent vectors and house IDs must have the same length"
        self.latent_vectors = latent_vectors.contiguous()
        self.house_ids = house_ids.contiguous()
        
    def __len__(self) -> int:
        return len(self.latent_vectors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latent_vectors[idx], self.house_ids[idx]


if __name__ == "__main__":
    import time
    
    DATA_DIRECTORY = './Ausgrid_processed_for_diffusion/per_house/'
    
    if os.path.exists(DATA_DIRECTORY):
        print("=== Testing Non-Overlapping Window DataLoader ===")
        
        start_time = time.time()
        # Create non-overlapping 2-day windows by setting step_size = window_size
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
            print(f"The second sample starts {dataset.sample_to_local_idx[1]} steps after the first.")
            
    else:
        print(f"ERROR: Data directory not found'{DATA_DIRECTORY}'")