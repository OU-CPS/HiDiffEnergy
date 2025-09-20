import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict


# --- Core Building Blocks ---

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timesteps into a vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock1D(nn.Module):
    """A standard ResNet block for 1D convolutional networks."""
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.block1_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block1_norm = nn.GroupNorm(8, out_channels)
        self.block1_act = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.block2_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.block2_norm = nn.GroupNorm(8, out_channels)
        self.block2_act = nn.SiLU()
        self.block2_dropout = nn.Dropout(dropout)

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.block1_act(self.block1_norm(self.block1_conv(x)))
        if self.time_mlp is not None and time_emb is not None:
            time_embedding = self.time_mlp(time_emb)
            h = h + time_embedding.unsqueeze(-1)
        h = self.block2_act(self.block2_norm(self.block2_conv(h)))
        h = self.block2_dropout(h)
        return h + self.res_conv(x)


class AttentionBlock1D(nn.Module):
    """Self-attention block for 1D sequences."""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, L)
        k = k.view(B, self.num_heads, C // self.num_heads, L)
        v = v.view(B, self.num_heads, C // self.num_heads, L)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)

        return x + self.proj(out)


class DownBlock1D(nn.Module):
    """Downsampling block for the U-Net encoder."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float, use_attention: bool, num_blocks: int = 2):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock1D(in_channels if i == 0 else out_channels, out_channels, time_emb_dim=time_emb_dim, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.attn = AttentionBlock1D(out_channels) if use_attention else nn.Identity()
        self.downsampler = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb):
        for resnet in self.resnets:
            x = resnet(x, time_emb)
        x = self.attn(x)
        return self.downsampler(x), x


class UpBlock1D(nn.Module):
    """Upsampling block for the U-Net decoder."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float, use_attention: bool, num_blocks: int = 2):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.resnets.append(ResnetBlock1D(in_channels * 2, out_channels, time_emb_dim=time_emb_dim, dropout=dropout))
        for _ in range(num_blocks - 1):
            self.resnets.append(ResnetBlock1D(out_channels, out_channels, time_emb_dim=time_emb_dim, dropout=dropout))
        self.attn = AttentionBlock1D(out_channels) if use_attention else nn.Identity()
        self.upsampler = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip_x, time_emb):
        x = self.upsampler(x)
        
        diff_L = skip_x.size()[-1] - x.size()[-1]
        x = F.pad(x, [diff_L // 2, diff_L - diff_L // 2])
        
        x = torch.cat([skip_x, x], dim=1)
        
        for resnet in self.resnets:
            x = resnet(x, time_emb)
        return self.attn(x)


class ConditionalUnet(nn.Module):
    def __init__(self, in_channels: int, num_houses: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [32, 64, 128, 256], 
                 dropout: float = 0.1, use_attention: bool = True, 
                 cond_channels: int = 0, blocks_per_level: int = 2):
        super().__init__()
        time_emb_dim = hidden_dims[0] * 4

        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dims[0]), nn.Linear(hidden_dims[0], time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        self.house_proj = nn.Linear(embedding_dim, time_emb_dim)

        self.day_of_week_embedding = nn.Embedding(7, embedding_dim) # 7 days
        self.day_of_year_embedding = nn.Embedding(366, embedding_dim) # 366 for leap years
        
        self.day_of_week_proj = nn.Linear(embedding_dim, time_emb_dim)
        self.day_of_year_proj = nn.Linear(embedding_dim, time_emb_dim)


        self.init_conv = nn.Conv1d(in_channels + cond_channels, hidden_dims[0], kernel_size=7, padding=3)
        
        num_resolutions = len(hidden_dims)
        self.down_blocks = nn.ModuleList([
            DownBlock1D(hidden_dims[i], hidden_dims[i+1], time_emb_dim, dropout, use_attention, blocks_per_level)
            for i in range(num_resolutions - 1)
        ])
        self.mid_block1 = ResnetBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim=time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock1D(hidden_dims[-1])
        self.mid_block2 = ResnetBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim=time_emb_dim, dropout=dropout)
        self.up_blocks = nn.ModuleList([
            UpBlock1D(hidden_dims[i+1], hidden_dims[i], time_emb_dim, dropout, use_attention, blocks_per_level)
            for i in reversed(range(num_resolutions - 1))
        ])
        self.final_conv = nn.Sequential(ResnetBlock1D(hidden_dims[0], hidden_dims[0], time_emb_dim=time_emb_dim, dropout=dropout), nn.Conv1d(hidden_dims[0], in_channels, 1))

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, conditions: Dict[str, torch.Tensor], conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_emb = self.time_mlp(timestep)
        
        house_id = conditions["house_id"]
        day_of_week = conditions["day_of_week"]
        day_of_year = conditions["day_of_year"]

        house_emb = self.house_proj(self.house_embedding(house_id))
        dow_emb = self.day_of_week_proj(self.day_of_week_embedding(day_of_week))
        doy_emb = self.day_of_year_proj(self.day_of_year_embedding(day_of_year))
        
        emb = time_emb + house_emb + dow_emb + doy_emb

        x = x.permute(0, 2, 1)
        if conditioning_signal is not None:
            x = torch.cat([x, conditioning_signal.permute(0, 2, 1)], dim=1)
        x = self.init_conv(x)
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip_x = down_block(x, emb)
            skip_connections.append(skip_x)
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, emb)), emb)
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), emb)
        return self.final_conv(x).permute(0, 2, 1)


class ImprovedDiffusionModel(nn.Module):
    """Wrapper for the U-Net model with diffusion logic."""
    def __init__(self, base_model: ConditionalUnet, num_timesteps: int):
        super().__init__()
        self.model = base_model
        self.num_timesteps = num_timesteps
        betas = self._cosine_beta_schedule(num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, axis=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1; x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999).float()

    def q_sample(self, x_start, t, noise):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_start + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise

    def forward(self, x_0: torch.Tensor, conditions: Dict[str, torch.Tensor], conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        predicted_noise = self.model(self.q_sample(x_0, t, noise), t, conditions, conditioning_signal)
        return F.huber_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, num_samples: int,
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        # Pass the whole conditions dictionary to the model
        predicted_noise = self.model(self.q_sample(x_0, t, noise), t, conditions, conditioning_signal)
        return F.huber_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, num_samples: int, conditions: Dict[str, torch.Tensor], shape: tuple, conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.randn(num_samples, *shape, device=device)
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps, leave=False):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Pass the whole conditions dictionary
            predicted_noise = self.model(x, t_batch, conditions, conditioning_signal)
            alpha_t, alpha_t_cumprod = 1.0 - self.betas[t], self.alphas_cumprod[t]
            x = (1 / alpha_t**0.5) * (x - (1 - alpha_t) / self.sqrt_one_minus_alphas_cumprod[t] * predicted_noise)
            if t > 0:
                variance = self.betas[t] * (1. - self.alphas_cumprod[t-1]) / (1. - alpha_t_cumprod)
                x += torch.sqrt(variance) * torch.randn_like(x)
        return x


class HierarchicalDiffusionModel(nn.Module):
    """
    Orchestrates a two-stage diffusion process where the fine model operates on
    smaller chunks of the full sequence.
    """
    def __init__(self, in_channels: int, num_houses: int, downscale_factor: int, **model_kwargs):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.fine_chunk_size = 2 * 96 

        num_timesteps = model_kwargs.pop("num_timesteps")
        
        self.downsampler = nn.AvgPool1d(kernel_size=downscale_factor)
        self.upsampler = nn.Upsample(scale_factor=downscale_factor, mode='linear', align_corners=False)

        self.coarse_model = ImprovedDiffusionModel(ConditionalUnet(in_channels=in_channels, num_houses=num_houses, **model_kwargs), num_timesteps)
        self.fine_model = ImprovedDiffusionModel(ConditionalUnet(in_channels=in_channels, num_houses=num_houses, cond_channels=in_channels, **model_kwargs), num_timesteps)

    # -- Change the forward signature to accept 'conditions' dictionary ---
    def forward(self, x_0: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Custom forward pass for efficient chunk-based training.
        """
        # 1. Coarse model training
        x_0_coarse = self.downsampler(x_0.permute(0, 2, 1)).permute(0, 2, 1)
        # --- FIX 2: Pass the entire 'conditions' dictionary ---
        coarse_loss = self.coarse_model(x_0_coarse, conditions)
        
        # 2. Fine model training
        with torch.no_grad():
            x_0_coarse_upsampled = self.upsampler(x_0_coarse.permute(0, 2, 1)).permute(0, 2, 1)
            x_0_fine_residual = x_0 - x_0_coarse_upsampled

        full_length = x_0.shape[1]
        start_index = torch.randint(0, full_length - self.fine_chunk_size + 1, (1,)).item()
        
        residual_chunk = x_0_fine_residual[:, start_index : start_index + self.fine_chunk_size, :]
        conditioning_chunk = x_0_coarse_upsampled[:, start_index : start_index + self.fine_chunk_size, :]
        
        # --- Pass the entire 'conditions' dictionary ---
        fine_loss = self.fine_model(residual_chunk, conditions, conditioning_signal=conditioning_chunk)

        return coarse_loss + fine_loss

    @torch.no_grad()
    # --- Change the sample signature to accept 'conditions' dictionary ---
    def sample(self, num_samples: int, conditions: Dict[str, torch.Tensor], shape: tuple) -> torch.Tensor:
        """
        Custom sample method that generates and stitches fine details in chunks.
        """
        full_length, num_features = shape
        
        # 1. Generate coarse structure
        print("--- Stage 1: Sampling Coarse Structure (Full duration) ---")
        coarse_shape = (full_length // self.downscale_factor, num_features)
        # - Pass the 'conditions' dictionary ---
        generated_coarse = self.coarse_model.sample(num_samples, conditions, shape=coarse_shape)
        upsampled_coarse = self.upsampler(generated_coarse.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 2. Iteratively generate fine details
        print("\n--- Stage 2: Iteratively Sampling Fine Details (2-Day Chunks) ---")
        stitched_fine_residual = torch.zeros_like(upsampled_coarse)
        
        for start_index in tqdm(range(0, full_length, self.fine_chunk_size), desc="Stitching fine chunks"):
            end_index = min(start_index + self.fine_chunk_size, full_length)
            chunk_length = end_index - start_index
            fine_shape = (chunk_length, num_features)
            conditioning_chunk = upsampled_coarse[:, start_index:end_index, :]
            
            # - Pass the 'conditions' dictionary ---
            generated_fine_chunk = self.fine_model.sample(
                num_samples, 
                conditions, 
                shape=fine_shape, 
                conditioning_signal=conditioning_chunk
            )
            
            stitched_fine_residual[:, start_index:end_index, :] = generated_fine_chunk

        # 3. Combine
        final_sample = upsampled_coarse + stitched_fine_residual
        return final_sample