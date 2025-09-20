import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

# Evaluation metrics
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from dtaidistance import dtw
from frechetdist import frdist
from dataloader import MultiHouseDataset
from diffusion_model import ImprovedDiffusionModel, ConditionalUnet

__version__ = "1.0.1"

# Setup device
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Using NVIDIA CUDA backend with optimizations.")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple MPS backend.")
else:
    DEVICE = "cpu"
    print("Using CPU.")

# --- Configuration ---
EPOCHS = 100
LEARNING_RATE = 3e-5
BATCH_SIZE = 512
USE_AMP = True
GRADIENT_CLIP_VAL = 1.0
WINDOW_DURATION = '7_days'
DATA_DIRECTORY = 'Ausgrid_processed_for_diffusion/per_house'
NUM_WORKERS = os.cpu_count() // 2
PIN_MEMORY = True
USE_ATTENTION = True
DROPOUT = 0.1
HIDDEN_SIZE = 256
EMBEDDING_DIM = 64
DIFFUSION_TIMESTEPS = 300


def denormalize_data(normalized_data, scaler_path='global_scaler.gz'):
    """Denormalize data from [-1, 1] back to the original scale."""
    scaler = joblib.load(scaler_path)
    original_shape = normalized_data.shape

    if len(original_shape) == 3:
        # Reshape 3D tensor for scaler, then revert
        batch_size, seq_len, features = original_shape
        normalized_flat = normalized_data.reshape(-1, features)
        denormalized_flat = scaler.inverse_transform(normalized_flat)
        return denormalized_flat.reshape(original_shape)
    else:
        # Handle 2D array
        return scaler.inverse_transform(normalized_data)


def calculate_window_size(duration: str) -> int:
    """Calculates window size in 30-min samples from a duration string."""
    SAMPLES_PER_DAY = 48
    if duration == '2_days': return 2 * SAMPLES_PER_DAY
    elif duration == '7_days': return 7 * SAMPLES_PER_DAY
    elif duration == '15_days': return 15 * SAMPLES_PER_DAY
    elif duration == '1_month': return 30 * SAMPLES_PER_DAY
    else: raise ValueError("Invalid WINDOW_DURATION.")


def moving_average(data, window_size):
    """Calculates the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def save_and_plot_loss(loss_dict, title, filepath, window_size=10):
    """Saves loss data to CSV and plots raw and smoothed loss curves."""
    plt.figure(figsize=(12, 6))
    for label, losses in loss_dict.items():
        pd.DataFrame({label: losses}).to_csv(f"{filepath}_{label.lower().replace(' ', '_')}.csv", index=False)
        plt.plot(losses, label=f'Raw {label}', alpha=0.3)
        if len(losses) > window_size:
            smoothed_losses = moving_average(losses, window_size)
            plt.plot(np.arange(window_size - 1, len(losses)), smoothed_losses, label=f'Smoothed {label}')
    plt.title(title)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(f"{filepath}.png"); plt.close()
    print(f"Loss plot saved to {filepath}.png")


def generate_and_plot_samples(model, num_samples, num_houses, num_features, device, filepath, window_size):
    """Generates and plots denormalized samples for visual inspection."""
    print("\nGenerating samples for visual inspection...")
    model.eval()
    sample_house_ids = torch.randint(0, num_houses, (num_samples,), device=device)
    with torch.no_grad():
        shape = (window_size, num_features)
        generated_data = model.sample(num_samples, sample_house_ids, shape=shape)
    generated_data_np = generated_data.cpu().numpy()
    generated_data_denorm = denormalize_data(generated_data_np)

    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        house_id = sample_house_ids[i].item()
        ax.plot(generated_data_denorm[i, :, 0], label='Grid Usage (kW)', color='dodgerblue')
        ax.plot(generated_data_denorm[i, :, 1], label='Solar Gen (kW)', color='darkorange', linestyle='--')
        ax.set_title(f'Generated Sample (House {house_id})')
        ax.set_ylabel('Power (kW)')
        ax.set_xlabel('Time (30-min intervals)')
        ax.grid(True, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    for j in range(num_samples, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Generated {window_size//48}-Day Profiles (Denormalized)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath); plt.close()
    print(f"Generated samples plot saved to {filepath}")


def evaluate_model(model, dataset, num_samples_to_eval, device, log_dir):
    """Runs a comprehensive evaluation of the model against a real dataset."""
    print("\nStarting comprehensive model evaluation...")
    model.eval()

    num_features = dataset[0][0].shape[1]
    window_size = dataset.window_size

    subset_indices = np.random.choice(len(dataset), num_samples_to_eval, replace=False)
    subset = Subset(dataset, subset_indices)
    real_dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

    real_samples = np.concatenate([batch[0].cpu().numpy() for batch, _ in real_dataloader], axis=0)

    print(f"Generating {num_samples_to_eval} samples for evaluation...")
    fake_house_ids = torch.randint(0, dataset.num_houses, (num_samples_to_eval,), device=device)
    with torch.no_grad():
        shape = (window_size, num_features)
        fake_samples = model.sample(num_samples_to_eval, fake_house_ids, shape=shape).cpu().numpy()

    print("Denormalizing samples for metric calculation...")
    real_samples_denorm = denormalize_data(real_samples)
    fake_samples_denorm = denormalize_data(fake_samples)

    report_path = os.path.join(log_dir, 'evaluation_report.txt')
    feature_names = ['Grid Usage (kW)', 'Solar Gen (kW)', 'Time (Sin)', 'Time (Cos)']

    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("      Diffusion Model Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write("NOTE: All metrics computed on DENORMALIZED data (original kW scale)\n\n")

        f.write("--- 1. Distributional Similarity (Kolmogorov-Smirnov Test) ---\n")
        ks_stats = []
        for i in range(2): # Grid and Solar only
            ks_stat, _ = ks_2samp(real_samples_denorm[:, :, i].flatten(), fake_samples_denorm[:, :, i].flatten())
            ks_stats.append(ks_stat)
            f.write(f"  {feature_names[i]}: KS statistic = {ks_stat:.4f}\n")
        f.write(f"\nAverage K-S Statistic (Grid & Solar only): {np.mean(ks_stats):.4f}\n\n")

        print("Calculating Autocorrelation...")
        acf_real_avg = np.mean([acf(real_samples_denorm[i, :, 0], nlags=48, fft=True) for i in range(min(500, num_samples_to_eval))], axis=0)
        acf_fake_avg = np.mean([acf(fake_samples_denorm[i, :, 0], nlags=48, fft=True) for i in range(min(500, num_samples_to_eval))], axis=0)
        acf_mse = np.mean((acf_real_avg - acf_fake_avg)**2)
        f.write("--- 2. Temporal Correlation (ACF Similarity) ---\n")
        f.write(f"ACF Similarity (MSE) for Grid Usage: {acf_mse:.6f}\n\n")

        plt.figure(figsize=(10, 6))
        plt.stem(acf_real_avg, linefmt='b-', markerfmt='bo', basefmt=' ', label='Real Data (Avg)')
        plt.stem(acf_fake_avg, linefmt='r-', markerfmt='ro', basefmt=' ', label='Generated Data (Avg)')
        plt.title('Average Autocorrelation - Grid Usage (Denormalized kW)')
        plt.xlabel('Lag (30-min intervals)'); plt.ylabel('Correlation')
        plt.legend(); plt.grid(True)
        acf_plot_path = os.path.join(log_dir, 'acf_comparison.png')
        plt.savefig(acf_plot_path); plt.close()
        print(f"ACF plot saved to {acf_plot_path}")

        f.write("--- 3. Trajectory Similarity (Denormalized) ---\n")
        print("Calculating trajectory similarity (DTW/Fréchet)...")
        dtw_distances_grid, dtw_distances_solar = [], []
        frechet_distances_grid, frechet_distances_solar = [], []
        for i in tqdm(range(min(500, num_samples_to_eval)), desc="DTW/Fréchet"):
            dtw_grid = dtw.distance(real_samples_denorm[i, :, 0], fake_samples_denorm[i, :, 0])
            frechet_grid = frdist(real_samples_denorm[i, :, :1], fake_samples_denorm[i, :, :1])
            dtw_solar = dtw.distance(real_samples_denorm[i, :, 1], fake_samples_denorm[i, :, 1])
            frechet_solar = frdist(real_samples_denorm[i, :, 1:2], fake_samples_denorm[i, :, 1:2])
            dtw_distances_grid.append(dtw_grid)
            dtw_distances_solar.append(dtw_solar)
            frechet_distances_grid.append(frechet_grid)
            frechet_distances_solar.append(frechet_solar)
        f.write(f"Dynamic Time Warping (DTW) Distance:\n")
        f.write(f"  Grid Usage:  Mean = {np.mean(dtw_distances_grid):.2f} kW, Std = {np.std(dtw_distances_grid):.2f} kW\n")
        f.write(f"  Solar Gen:   Mean = {np.mean(dtw_distances_solar):.2f} kW, Std = {np.std(dtw_distances_solar):.2f} kW\n\n")
        f.write(f"Fréchet Distance:\n")
        f.write(f"  Grid Usage:  Mean = {np.mean(frechet_distances_grid):.2f} kW, Std = {np.std(frechet_distances_grid):.2f} kW\n")
        f.write(f"  Solar Gen:   Mean = {np.mean(frechet_distances_solar):.2f} kW, Std = {np.std(frechet_distances_solar):.2f} kW\n\n")

        f.write("--- 4. Basic Statistics Comparison (Denormalized) ---\n")
        for i in range(2):
            real_mean, real_std = np.mean(real_samples_denorm[:, :, i]), np.std(real_samples_denorm[:, :, i])
            fake_mean, fake_std = np.mean(fake_samples_denorm[:, :, i]), np.std(fake_samples_denorm[:, :, i])
            f.write(f"{feature_names[i]}:\n")
            f.write(f"  Real: Mean = {real_mean:.3f} kW, Std = {real_std:.3f} kW\n")
            f.write(f"  Fake: Mean = {fake_mean:.3f} kW, Std = {fake_std:.3f} kW\n")
            f.write(f"  Difference: Mean = {abs(real_mean - fake_mean):.3f} kW, Std = {abs(real_std - fake_std):.3f} kW\n\n")

    print(f"Evaluation complete. Report saved to {report_path}")


def train_diffusion(log_dir, model_save_path):
    """Main training loop for the conditional diffusion model."""
    print("--- Starting Direct Conditional Diffusion Training ---")
    window_size = calculate_window_size(WINDOW_DURATION)
    print(f"Using window duration: {WINDOW_DURATION} ({window_size} samples)")

    dataset = MultiHouseDataset(data_dir=DATA_DIRECTORY, window_size=window_size, step_size=window_size)
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_houses} houses, {dataset[0][0].shape[1]} features.")

    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    base_model = ConditionalUnet(
        in_channels=dataset[0][0].shape[1],
        num_houses=dataset.num_houses,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[HIDDEN_SIZE // 4, HIDDEN_SIZE // 2, HIDDEN_SIZE],
        dropout=DROPOUT,
        use_attention=USE_ATTENTION
    )
    model = ImprovedDiffusionModel(base_model=base_model, num_timesteps=DIFFUSION_TIMESTEPS).to(DEVICE)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile for performance.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
        for clean_data, house_id in pbar:
            clean_data = clean_data.to(DEVICE, non_blocking=PIN_MEMORY)
            house_id = house_id.to(DEVICE, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                loss = model(clean_data, house_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            scaler.step(optimizer); scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for clean_data, house_id in tqdm(val_dataloader, desc="Validating"):
                clean_data = clean_data.to(DEVICE, non_blocking=PIN_MEMORY)
                house_id = house_id.to(DEVICE, non_blocking=PIN_MEMORY)
                with autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                    loss = model(clean_data, house_id)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")
        scheduler.step()

    print("--- Training complete ---")
    save_and_plot_loss(
        {'Train Loss': train_losses, 'Validation Loss': val_losses},
        'Diffusion Model Training & Validation Loss',
        os.path.join(log_dir, 'diffusion_loss_curves')
    )
    return dataset


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"diffusion_{WINDOW_DURATION}_{timestamp}"
    log_dir = os.path.join("training_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, 'best_diffusion_model.pth')

    print(f"Starting new run: {run_name}")
    print(f"Logs and models will be saved to: {log_dir}")

    full_dataset = train_diffusion(log_dir=log_dir, model_save_path=model_path)

    num_houses = full_dataset.num_houses
    num_features = full_dataset[0][0].shape[1]
    window_size = full_dataset.window_size

    base_model = ConditionalUnet(
        in_channels=num_features,
        num_houses=num_houses,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[HIDDEN_SIZE // 4, HIDDEN_SIZE // 2, HIDDEN_SIZE],
        dropout=DROPOUT,
        use_attention=USE_ATTENTION
    )
    final_model = ImprovedDiffusionModel(base_model=base_model, num_timesteps=DIFFUSION_TIMESTEPS).to(DEVICE)

    # Load the best model's state dict for final evaluation
    saved_state_dict = torch.load(model_path, map_location=DEVICE)
    cleaned_state_dict = {}
    for key, value in saved_state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value
    final_model.load_state_dict(cleaned_state_dict)

    if hasattr(torch, 'compile'):
        final_model = torch.compile(final_model)

    generate_and_plot_samples(
        model=final_model,
        num_samples=8,
        num_houses=num_houses,
        num_features=num_features,
        device=DEVICE,
        filepath=os.path.join(log_dir, 'final_generated_samples.png'),
        window_size=window_size
    )

    evaluate_model(
        model=final_model,
        dataset=full_dataset,
        num_samples_to_eval=2000,
        device=DEVICE,
        log_dir=log_dir
    )