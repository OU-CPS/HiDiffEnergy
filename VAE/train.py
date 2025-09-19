import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from dtaidistance import dtw

# --- Custom Modules ---
from autoencoder import ImprovedConditionalVAE as ConditionalVAE
from dataloader import MultiHouseDataset

# --- Environment Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
print(f"Using device: {DEVICE}")

# --- Training Parameters ---
EPOCHS = 150
LEARNING_RATE = 1e-5
BATCH_SIZE = 4096
USE_AMP = True
GRADIENT_CLIP_VAL = 1.0
DATA_DIRECTORY = 'Ausgrid_processed_for_diffusion/per_house'
NUM_WORKERS = os.cpu_count() // 2
PIN_MEMORY = True

# --- Model & Data Configuration ---
WINDOW_DURATION = '30_days' # Options: '2_days', '7_days', '30_days'
LATENT_DIM = 64

# --- VAE-Specific Parameters ---
BETA_MAX = 1.0
ANNEAL_METHOD = "cyclical"
ANNEAL_CYCLES = 4
FREE_BITS_LAMBDA = 0.05 # Free bits budget to prevent posterior collapse
DECODER_DROPOUT = 0.2  # Decoder dropout rate

# --- Helper Functions ---

def calculate_window_size(duration: str) -> int:
    """Calculates window size from duration string."""
    SAMPLES_PER_DAY = 48
    duration_map = {'2_days': 2, '7_days': 7, '15_days': 15, '30_days': 30}
    if duration in duration_map:
        return duration_map[duration] * SAMPLES_PER_DAY
    raise ValueError("Invalid WINDOW_DURATION.")

def save_and_plot_loss(loss_dict, title, filepath):
    """Saves loss data to CSV and plots the curves."""
    plt.figure(figsize=(12, 6))
    for label, losses in loss_dict.items():
        pd.DataFrame({label: losses}).to_csv(f"{filepath}_{label.lower().replace(' ', '_')}.csv", index=False)
        plt.plot(losses, label=label)
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(f"{filepath}.png"); plt.close()
    print(f"Loss plot saved to {filepath}.png")

def generate_and_plot_samples(model, num_samples, num_houses, device, filepath, window_size):
    """Generates and plots VAE samples."""
    model.eval()
    sample_house_ids = torch.randint(0, num_houses, (num_samples,), device=device)
    with torch.no_grad():
        generated_data = model.sample(num_samples, sample_house_ids, device).cpu().numpy()

    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(num_samples):
        axes[i].plot(generated_data[i, :, 0], label='Grid Usage')
        axes[i].plot(generated_data[i, :, 1], label='Solar Gen', linestyle='--')
        axes[i].set_title(f'Generated (House {sample_house_ids[i].item()})')
        axes[i].grid(True, alpha=0.5)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    for j in range(num_samples, len(axes)): axes[j].set_visible(False)
    plt.suptitle(f'Generated {window_size//48}-Day Profiles (VAE)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath); plt.close()
    print(f"Generated samples plot saved to {filepath}")

# --- VAE Loss & Annealing ---

def vae_loss_function(recon_x, x, mu, log_var, beta, free_bits_lambda):
    """Calculates VAE loss with the Free Bits technique."""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    kl_loss = torch.clamp(kl_div, min=free_bits_lambda) # Apply free bits
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_div # Return original KL for logging

def get_kl_annealing_factor(epoch, total_epochs, method, num_cycles):
    """Calculates the KL annealing factor (beta)."""
    if method == "monotonic":
        return min(1.0, epoch / (total_epochs * 0.25))
    elif method == "cyclical":
        cycle_len = total_epochs / num_cycles
        return min(1.0, 2 * ((epoch % cycle_len) / cycle_len))
    return 1.0

# --- Standardized Evaluation ---

def mmd_rbf(x, y, sigma=1.0):
    """Computes Maximum Mean Discrepancy (MMD) with a Gaussian RBF kernel."""
    x, y = torch.from_numpy(x).float().to(DEVICE), torch.from_numpy(y).float().to(DEVICE)
    if x.dim() == 1: x = x.unsqueeze(1)
    if y.dim() == 1: y = y.unsqueeze(1)
    beta = 1. / (2. * sigma**2)
    dist = torch.cdist(x, y, p=2).pow(2)
    kernel = torch.exp(-beta * dist)
    return kernel.mean()

def evaluate_model(model, dataset, num_samples, device, log_dir):
    """Performs a comprehensive evaluation of the model."""
    print("\n--- Starting Model Evaluation ---")
    model.eval()

    # Prepare real and generated data
    subset_indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_loader = DataLoader(Subset(dataset, subset_indices), batch_size=BATCH_SIZE, shuffle=False)
    real_samples = np.concatenate([b[0].cpu().numpy() for b in real_loader], axis=0)
    
    fake_house_ids = torch.randint(0, dataset.num_houses, (num_samples,), device=device)
    with torch.no_grad():
        fake_samples = model.sample(num_samples, fake_house_ids, device).cpu().numpy()

    report_path = os.path.join(log_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*25 + " VAE Evaluation Report " + "="*25 + "\n\n")
        
        # K-S Test
        ks_stats = [ks_2samp(real_samples[:, :, i].flatten(), fake_samples[:, :, i].flatten())[0] for i in range(real_samples.shape[2])]
        f.write(f"1. K-S Test (Distribution): {np.mean(ks_stats):.4f} (Lower is better)\n")

        # MMD Test
        mmd_scores = []
        for i in tqdm(range(real_samples.shape[2]), desc="MMD", leave=False):
            real_flat, fake_flat = real_samples[:, :, i].flatten(), fake_samples[:, :, i].flatten()
            batch_mmds = [mmd_rbf(np.random.choice(real_flat, 2048), np.random.choice(fake_flat, 2048)).item() for _ in range(100)]
            mmd_scores.append(np.mean(batch_mmds))
        f.write(f"2. MMD (Distribution): {np.mean(mmd_scores):.6f} (Lower is better)\n")
        
        # ACF Similarity
        acf_real = np.mean([acf(s[:, 0], nlags=48, fft=True) for s in real_samples], axis=0)
        acf_fake = np.mean([acf(s[:, 0], nlags=48, fft=True) for s in fake_samples], axis=0)
        acf_mse = np.mean((acf_real - acf_fake)**2)
        f.write(f"3. ACF MSE (Temporal): {acf_mse:.6f} (Lower is better)\n")

        # DTW
        dtw_dists = [(dtw.distance(r[:, 0], f[:, 0]) + dtw.distance(r[:, 1], f[:, 1])) / 2.0 
                     for r, f in zip(tqdm(real_samples[:500], desc="DTW", leave=False), fake_samples[:500])]
        f.write(f"4. DTW (Shape): Mean={np.mean(dtw_dists):.4f}, Std={np.std(dtw_dists):.4f} (Lower is better)\n")

    print(f"Evaluation complete. Report saved to {report_path}")

# --- Training Loop ---

def train_vae(log_dir, model_save_path):
    print("--- Training Conditional VAE ---")
    window_size = calculate_window_size(WINDOW_DURATION)
    dataset = MultiHouseDataset(data_dir=DATA_DIRECTORY, window_size=window_size, step_size=window_size//10, limit_to_one_year=False)
    
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ConditionalVAE(
        in_channels=dataset[0][0].shape[1], 
        num_houses=dataset.num_houses,  
        window_size=window_size, 
        latent_dim=LATENT_DIM,
        dropout_rate=DECODER_DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    losses = {'Train': [], 'Val': [], 'KL Div': []}
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_kl_div = 0, 0
        beta = get_kl_annealing_factor(epoch, EPOCHS, ANNEAL_METHOD, ANNEAL_CYCLES) * BETA_MAX
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for data, house_id in pbar:
            data, house_id = data.to(DEVICE), house_id.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                recon, mu, log_var = model(data, house_id)
                loss, recon_loss, kl_div = vae_loss_function(recon, data, mu, log_var, beta, FREE_BITS_LAMBDA)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            scaler.step(optimizer); scaler.update()
            total_train_loss += loss.item(); total_kl_div += kl_div.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'kl': f'{kl_div.item():.4f}'})

        losses['Train'].append(total_train_loss / len(train_loader))
        losses['KL Div'].append(total_kl_div / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, house_id in val_loader:
                data, house_id = data.to(DEVICE), house_id.to(DEVICE)
                recon, mu, log_var = model(data, house_id)
                loss, _, _ = vae_loss_function(recon, data, mu, log_var, 1.0, 0.0) # Beta=1, no free bits for val
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        losses['Val'].append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {losses['Train'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved (Val Loss: {best_val_loss:.4f})")

    save_and_plot_loss(losses, 'VAE Training Curves', os.path.join(log_dir, 'vae_training_curves'))
    return dataset

# --- Main Execution ---
if __name__ == "__main__":
    run_name = f"vae_{WINDOW_DURATION}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    log_dir = os.path.join("training_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, 'best_vae_model.pth')

    print(f"Starting VAE run: {run_name}\nLogs will be saved to: {log_dir}")
    
    full_dataset = train_vae(log_dir, model_path)
    
    print("\nLoading best model for final evaluation...")
    final_model = ConditionalVAE(
        in_channels=full_dataset[0][0].shape[1],
        num_houses=full_dataset.num_houses,
        window_size=full_dataset.window_size,
        latent_dim=LATENT_DIM,
        dropout_rate=DECODER_DROPOUT
    ).to(DEVICE)
    final_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    generate_and_plot_samples(final_model, 8, full_dataset.num_houses, DEVICE, os.path.join(log_dir, 'final_samples.png'), full_dataset.window_size)
    evaluate_model(final_model, full_dataset, 2000, DEVICE, log_dir)
