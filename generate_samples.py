import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Ensure this script is in the same directory as your other files
from dataloader import MultiHouseDataset
from hierarchial_diffusion_model import HierarchicalDiffusionModel

# Path to the trained model file from your training run
MODEL_PATH = 'training_logs/hierarchical_diffusion_2_days_2025-09-22_15-50-30/best_hierarchical_model.pth' 

HOUSE_IDS_TO_USE = []

# How many different time-series samples to generate for each house
NUM_SAMPLES_PER_HOUSE = 100

# Where to save the final comparison plot
OUTPUT_PLOT_PATH = 'comparison_plot.png'

# These MUST exactly match the parameters used to train the model specified in MODEL_PATH
WINDOW_DURATION = '2_days'
HIDDEN_SIZE = 256
EMBEDDING_DIM = 64
DIFFUSION_TIMESTEPS = 500
DOWNSCALE_FACTOR = 4
DROPOUT = 0.1
USE_ATTENTION = True
BLOCKS_PER_LEVEL = 4

DATA_DIRECTORY = 'Ausgrid_processed_for_diffusion/per_house'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_window_size(duration: str) -> int:
    SAMPLES_PER_DAY = 48
    mapping = {
        '2_days': 2 * SAMPLES_PER_DAY, '7_days': 7 * SAMPLES_PER_DAY,
        '15_days': 15 * SAMPLES_PER_DAY, '30_days': 30 * SAMPLES_PER_DAY
    }
    return mapping[duration]

def denormalize_data(normalized_data, scaler_path='global_scaler.gz'):
    scaler = joblib.load(scaler_path)
    original_shape = normalized_data.shape
    if len(original_shape) == 3:
        b, s, f = original_shape
        flat = normalized_data.reshape(-1, f)
        denorm_flat = scaler.inverse_transform(flat)
        return denorm_flat.reshape(original_shape)
    else:
        return scaler.inverse_transform(normalized_data)


def load_model_for_inference(model_path, num_features, num_houses):
    """Loads the trained model weights into a new model instance."""
    print(f"Loading model from: {model_path}")
    
    model = HierarchicalDiffusionModel(
        in_channels=num_features,
        num_houses=num_houses,
        downscale_factor=DOWNSCALE_FACTOR,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[HIDDEN_SIZE // 4, HIDDEN_SIZE // 2, HIDDEN_SIZE],
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        num_timesteps=DIFFUSION_TIMESTEPS,
        blocks_per_level=BLOCKS_PER_LEVEL
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value
            
    model.load_state_dict(cleaned_state_dict)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
    return model

def generate_synthetic_data(model, house_ids, num_samples_per_house, window_size, num_features):
    """Generates and denormalizes data for a list of specified houses."""
    print(f"Generating {num_samples_per_house} sample(s) for houses: {house_ids}")
    generated_data = {}
    
    with torch.no_grad():
        for house_id in tqdm(house_ids, desc="Generating for each house"):
            conditions = {
                "house_id": torch.tensor([house_id] * num_samples_per_house, device=DEVICE),
                "day_of_week": torch.randint(0, 7, (num_samples_per_house,), device=DEVICE),
                "day_of_year": torch.randint(0, 365, (num_samples_per_house,), device=DEVICE)
            }
            shape = (window_size, num_features)
            
            samples_norm = model.sample(num_samples_per_house, conditions, shape=shape)
            
            samples_denorm = denormalize_data(samples_norm.cpu().numpy())
            generated_data[house_id] = samples_denorm
            
    return generated_data

def fetch_real_data(house_ids, num_samples_per_house, dataset):
    """Finds and denormalizes real data samples for the specified houses."""
    print(f"Fetching {num_samples_per_house} real sample(s) for houses: {house_ids}")
    real_data = {house_id: [] for house_id in house_ids}
    
    for i in tqdm(range(len(dataset)), desc="Searching for real samples"):
        _, conditions = dataset[i]
        house_id = conditions["house_id"].item()
        
        if house_id in house_ids and len(real_data[house_id]) < num_samples_per_house:
            sample_norm, _ = dataset[i]
            sample_denorm = denormalize_data(sample_norm.unsqueeze(0).numpy())
            real_data[house_id].append(sample_denorm[0])

        if all(len(samples) >= num_samples_per_house for samples in real_data.values()):
            break
            
    for house_id in real_data:
        real_data[house_id] = np.array(real_data[house_id])
        
    return real_data

def plot_comparison(generated_data, real_data, output_path, window_duration):
    """Creates an overlapped, side-by-side plot of generated vs. real data."""
    print(f"Creating comparison plot and saving to {output_path}")
    num_houses = len(generated_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_houses))
    
    for i, (house_id, samples) in enumerate(generated_data.items()):
        for sample in samples:
            ax1.plot(sample[:, 0], color=colors[i], alpha=0.8, label=f'House {house_id}' if i < 5 else None) # Grid
            ax1.plot(sample[:, 1], color=colors[i], alpha=0.8, linestyle='--') # Solar

    ax1.set_title(f'Generated Data ({num_houses} Overlapped Houses)', fontsize=16)
    ax1.set_xlabel('Time (30-min intervals)')
    ax1.set_ylabel('Power (kW)')
    ax1.grid(True, alpha=0.5)
    
    for i, (house_id, samples) in enumerate(real_data.items()):
        for sample in samples:
            ax2.plot(sample[:, 0], color=colors[i], alpha=0.8) # Grid
            ax2.plot(sample[:, 1], color=colors[i], alpha=0.8, linestyle='--') # Solar

    ax2.set_title(f'Real Data ({num_houses} Overlapped Houses)', fontsize=16)
    ax2.set_xlabel('Time (30-min intervals)')
    ax2.grid(True, alpha=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Grid Usage'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Solar Gen')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.suptitle(f'Comparison of {window_duration} Generated vs. Real Load Profiles', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print("Plot saved successfully.")


if __name__ == "__main__":
    window_size = calculate_window_size(WINDOW_DURATION)
    full_dataset = MultiHouseDataset(
        data_dir=DATA_DIRECTORY, 
        window_size=window_size,
        step_size=window_size
    )
    num_features = full_dataset[0][0].shape[1]
    num_houses_total = full_dataset.num_houses
    print(f"Dataset contains {num_houses_total} houses in total.")

    model = load_model_for_inference(MODEL_PATH, num_features, num_houses_total)
    
    synthetic_data = generate_synthetic_data(
        model, HOUSE_IDS_TO_USE, NUM_SAMPLES_PER_HOUSE, window_size, num_features
    )
    
    real_data_samples = fetch_real_data(HOUSE_IDS_TO_USE, NUM_SAMPLES_PER_HOUSE, full_dataset)
    
    plot_comparison(synthetic_data, real_data_samples, OUTPUT_PLOT_PATH, WINDOW_DURATION)
    
    print("\nScript finished.")

