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
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from dtaidistance import dtw
from dataloader import MultiHouseDataset
from hierarchial_diffusion_model import HierarchicalDiffusionModel

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

EPOCHS = 500
LEARNING_RATE = 1e-5
BATCH_SIZE = 1024
USE_AMP = True
GRADIENT_CLIP_VAL = 1.0
WINDOW_DURATION = '15_days'
DATA_DIRECTORY = 'Ausgrid_processed_for_diffusion/per_house'
NUM_WORKERS = os.cpu_count() // 2
PIN_MEMORY = True
USE_ATTENTION = True
DROPOUT = 0.1
HIDDEN_SIZE = 256
EMBEDDING_DIM = 64
DIFFUSION_TIMESTEPS = 600
DOWNSCALE_FACTOR = 4

def calculate_window_size(duration: str) -> int:
    SAMPLES_PER_DAY = 48
    if duration == '2_days': return 2 * SAMPLES_PER_DAY
    elif duration == '7_days': return 7 * SAMPLES_PER_DAY
    elif duration == '15_days': return 15 * SAMPLES_PER_DAY
    elif duration == '30_days': return 30 * SAMPLES_PER_DAY
    elif duration == '1_month': return 30 * SAMPLES_PER_DAY
    else: raise ValueError("Invalid WINDOW_DURATION.")

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def save_and_plot_loss(loss_dict, title, filepath, window_size=10):
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
    print("Generating samples for visual inspection")
    model.eval()
    
    sample_conditions = {
        "house_id": torch.randint(0, num_houses, (num_samples,), device=device),
        "day_of_week": torch.randint(0, 7, (num_samples,), device=device),
        "day_of_year": torch.randint(0, 365, (num_samples,), device=device)
    }

    with torch.no_grad():
        shape = (window_size, num_features)
        generated_data = model.sample(num_samples, sample_conditions, shape=shape)
    
    generated_data_np = generated_data.cpu().numpy()

    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(num_samples):
        ax = axes[i]
        house_id = sample_conditions["house_id"][i].item()
        ax.plot(generated_data_np[i, :, 0], label='Grid Usage', color='dodgerblue')
        ax.plot(generated_data_np[i, :, 1], label='Solar Gen', color='darkorange', linestyle='--')
        ax.set_title(f'Generated Sample (House {house_id})')
        ax.grid(True, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    for j in range(num_samples, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Generated {window_size//48}-Day Profiles', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath); plt.close()
    print(f"Generated samples plot saved to {filepath}")

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1. / (2. * sigma**2)
    dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist)

def mmd_rbf(x, y, sigma=1.0):
    x = torch.from_numpy(x).float().to(DEVICE)
    y = torch.from_numpy(y).float().to(DEVICE)
    
    if x.dim() == 1: x = x.unsqueeze(1)
    if y.dim() == 1: y = y.unsqueeze(1)

    xx = gaussian_kernel(x, x, sigma).mean()
    yy = gaussian_kernel(y, y, sigma).mean()
    xy = gaussian_kernel(x, y, sigma).mean()
    
    return xx + yy - 2 * xy

def evaluate_model(model, dataset, num_samples_to_eval, device, log_dir):
    print("Starting Comprehensive Model Evaluation")
    model.eval()

    num_features = dataset[0][0].shape[1]
    window_size = dataset.window_size
    num_houses = dataset.num_houses
    
    subset_indices = np.random.choice(len(dataset), num_samples_to_eval, replace=False)
    subset = Subset(dataset, subset_indices)
    real_dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    
    real_samples_list = []
    for batch, _ in real_dataloader:
        real_samples_list.append(batch.cpu().numpy())
    real_samples = np.concatenate(real_samples_list, axis=0)

    print(f"Generating {num_samples_to_eval} samples for evaluation")
    fake_conditions = {
        "house_id": torch.randint(0, num_houses, (num_samples_to_eval,), device=device),
        "day_of_week": torch.randint(0, 7, (num_samples_to_eval,), device=device),
        "day_of_year": torch.randint(0, 365, (num_samples_to_eval,), device=device)
    }
    with torch.no_grad():
        shape = (window_size, num_features)
        fake_samples = model.sample(num_samples_to_eval, fake_conditions, shape=shape).cpu().numpy()

    report_path = os.path.join(log_dir, 'evaluation_report.txt')
    feature_names = ['Grid Usage', 'Solar Gen', 'Time (Sin)', 'Time (Cos)']

    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("      Hierarchical Diffusion Model Evaluation Report\n")
        f.write("="*50 + "\n\n")

        f.write("--- 1. Distributional Similarity (Kolmogorov-Smirnov Test) ---\n")
        f.write("Measures the maximum difference between cumulative distribution functions. Lower is better.\n\n")
        ks_stats = []
        for i in range(num_features):
            ks_stat, _ = ks_2samp(real_samples[:, :, i].flatten(), fake_samples[:, :, i].flatten())
            ks_stats.append(ks_stat)
        avg_ks_stat = np.mean(ks_stats)
        f.write(f"Average K-S Statistic (across all features): {avg_ks_stat:.4f}\n\n")
        
        print("Calculating Maximum Mean Discrepancy (MMD)")
        f.write("--- 2. Distributional Similarity (Maximum Mean Discrepancy) ---\n")
        f.write("Measures the distance between distributions in a reproducing kernel Hilbert space. Lower is better.\n\n")
        
        MMD_BATCH_SIZE = 2048
        N_BATCHES = 100

        mmd_scores = []
        for i in range(num_features):
            real_flat = real_samples[:, :, i].flatten()
            fake_flat = fake_samples[:, :, i].flatten()
            
            batch_mmds = []
            for _ in tqdm(range(N_BATCHES), desc=f"MMD Feature {i+1}/{num_features}", leave=False):
                real_batch = np.random.choice(real_flat, MMD_BATCH_SIZE, replace=True)
                fake_batch = np.random.choice(fake_flat, MMD_BATCH_SIZE, replace=True)
                mmd_val = mmd_rbf(real_batch, fake_batch)
                batch_mmds.append(mmd_val.item())
            
            mmd_scores.append(np.mean(batch_mmds))
            
        avg_mmd_score = np.mean(mmd_scores)
        f.write(f"Average MMD Score (across all features): {avg_mmd_score:.6f}\n\n")

        print("Calculating Autocorrelation")
        acf_real_avg = np.mean([acf(real_samples[i, :, 0], nlags=48, fft=True) for i in range(num_samples_to_eval)], axis=0)
        acf_fake_avg = np.mean([acf(fake_samples[i, :, 0], nlags=48, fft=True) for i in range(num_samples_to_eval)], axis=0)
        acf_mse = np.mean((acf_real_avg - acf_fake_avg)**2)
        
        f.write("--- 3. Temporal Correlation (ACF Similarity) ---\n")
        f.write("Measures the MSE between the average ACFs of real and generated data. Lower values indicate better temporal realism.\n\n")
        f.write(f"ACF Similarity (MSE): {acf_mse:.6f}\n\n")

        plt.figure(figsize=(10, 6))
        plt.stem(acf_real_avg, linefmt='b-', markerfmt='bo', basefmt=' ', label='Real Data (Avg)')
        plt.stem(acf_fake_avg, linefmt='r-', markerfmt='ro', basefmt=' ', label='Generated Data (Avg)')
        plt.title('Average Autocorrelation (Grid Usage)')
        plt.xlabel('Lag (30-min intervals)'); plt.ylabel('Correlation')
        plt.legend(); plt.grid(True)
        acf_plot_path = os.path.join(log_dir, 'acf_comparison.png')
        plt.savefig(acf_plot_path); plt.close()
        print(f"ACF plot saved to {acf_plot_path}")

        f.write("--- 4. Trajectory Similarity (DTW) ---\n")
        f.write("Measures the shape similarity between real and generated series. Lower is better.\n\n")
        
        print("Calculating trajectory similarity")
        dtw_distances = []
        
        for i in tqdm(range(min(500, num_samples_to_eval)), desc="DTW"):
            dtw_grid = dtw.distance(real_samples[i, :, 0], fake_samples[i, :, 0])
            dtw_solar = dtw.distance(real_samples[i, :, 1], fake_samples[i, :, 1])
            dtw_distances.append((dtw_grid + dtw_solar) / 2.0)
            
        f.write(f"Dynamic Time Warping (DTW) Distance (Avg over Grid & Solar):\n")
        f.write(f"  - Mean: {np.mean(dtw_distances):.4f}\n")
        f.write(f"  - Std Dev: {np.std(dtw_distances):.4f}\n\n")

    print(f"Evaluation complete. Report saved to {report_path}")

def train_diffusion(log_dir, model_save_path):
    print("Starting Hierarchical Diffusion Training")
    window_size = calculate_window_size(WINDOW_DURATION)
    print(f"Using window duration: {WINDOW_DURATION} ({window_size} samples)")

    dataset = MultiHouseDataset(data_dir=DATA_DIRECTORY, window_size=window_size, step_size=96, limit_to_one_year=False)
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_houses} houses, {dataset[0][0].shape[1]} features.")

    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = HierarchicalDiffusionModel(
        in_channels=dataset[0][0].shape[1],
        num_houses=dataset.num_houses,
        downscale_factor=DOWNSCALE_FACTOR,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[HIDDEN_SIZE // 4, HIDDEN_SIZE // 2, HIDDEN_SIZE],
        dropout=DROPOUT, 
        use_attention=USE_ATTENTION,
        num_timesteps=DIFFUSION_TIMESTEPS,
        blocks_per_level=3
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"Starting training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")
        
        for clean_data, conditions in pbar:
            clean_data = clean_data.to(DEVICE, non_blocking=PIN_MEMORY)
            conditions = {k: v.to(DEVICE, non_blocking=PIN_MEMORY) for k, v in conditions.items()}
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                loss = model(clean_data, conditions)

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
            for clean_data, conditions in tqdm(val_dataloader, desc="Validating"):
                clean_data = clean_data.to(DEVICE, non_blocking=PIN_MEMORY)
                conditions = {k: v.to(DEVICE, non_blocking=PIN_MEMORY) for k, v in conditions.items()}
                with autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                    loss = model(clean_data, conditions)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")
        scheduler.step()

    print("Training complete")
    save_and_plot_loss(
        {'Train Loss': train_losses, 'Validation Loss': val_losses},
        'Hierarchical Diffusion Model Training & Validation Loss',
        os.path.join(log_dir, 'diffusion_loss_curves')
    )
    
    return dataset

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"hierarchical_diffusion_{WINDOW_DURATION}_{timestamp}"
    log_dir = os.path.join("training_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, 'best_hierarchical_model.pth')

    print(f"Starting new run: {run_name}")
    print(f"Logs and models will be saved to: {log_dir}")

    full_dataset = train_diffusion(log_dir=log_dir, model_save_path=model_path)
    
    sample_data, _ = full_dataset[0]
    num_features = sample_data.shape[1]
    
    num_houses = full_dataset.num_houses
    window_size = full_dataset.window_size

    final_model = HierarchicalDiffusionModel(
        in_channels=num_features,
        num_houses=num_houses,
        downscale_factor=DOWNSCALE_FACTOR,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[HIDDEN_SIZE // 4, HIDDEN_SIZE // 2, HIDDEN_SIZE],
        dropout=DROPOUT, 
        use_attention=USE_ATTENTION,
        num_timesteps=DIFFUSION_TIMESTEPS,
        blocks_per_level=3 
    ).to(DEVICE)
    
    saved_state_dict = torch.load(model_path, map_location=DEVICE)
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in saved_state_dict.items()}
    final_model.load_state_dict(cleaned_state_dict)

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
