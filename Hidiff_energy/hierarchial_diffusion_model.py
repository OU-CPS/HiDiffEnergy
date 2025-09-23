import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict
from tqdm import tqdm


class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        ) if time_emb_dim is not None else None

        self.block1_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block1_norm = nn.GroupNorm(8, out_channels, affine=False)
        self.block1_act = nn.SiLU()

        self.block2_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.block2_norm = nn.GroupNorm(8, out_channels)
        self.block2_act = nn.SiLU()
        self.block2_dropout = nn.Dropout(dropout)

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.block1_conv(x)
        h = self.block1_norm(h)

        if self.time_mlp is not None and time_emb is not None:
            scale_shift = self.time_mlp(time_emb)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (scale.unsqueeze(-1) + 1) + shift.unsqueeze(-1)

        h = self.block1_act(h)

        h = self.block2_act(self.block2_norm(self.block2_conv(h)))
        h = self.block2_dropout(h)
        return h + self.res_conv(x)


class AttentionBlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        out = out.permute(0, 1, 3, 2)
        out = out.contiguous().view(B, C, L)
        
        return x + self.proj(out)


class DownBlock1D(nn.Module):
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
        skip = x
        x = self.downsampler(x)
        return x, skip


class UpBlock1D(nn.Module):
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
        
        if x.size(-1) != skip_x.size(-1):
            diff_L = skip_x.size(-1) - x.size(-1)
            if diff_L > 0:
                x = F.pad(x, [diff_L // 2, diff_L - diff_L // 2])
            elif diff_L < 0:
                x = x[:, :, :skip_x.size(-1)]
        
        x = torch.cat([skip_x, x], dim=1)
        
        for resnet in self.resnets:
            x = resnet(x, time_emb)
        return self.attn(x)


class ConditionalUnet(nn.Module):
    def __init__(self, in_channels: int, num_houses: int, embedding_dim: int = 64, 
                 hidden_dims: List[int] = [64, 128, 256], 
                 dropout: float = 0.1, use_attention: bool = True, 
                 cond_channels: int = 0, blocks_per_level: int = 2):
        super().__init__()
        time_emb_dim = hidden_dims[0] * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]), 
            nn.Linear(hidden_dims[0], time_emb_dim), 
            nn.SiLU(), 
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        self.house_proj = nn.Linear(embedding_dim, time_emb_dim)

        self.day_of_week_embedding = nn.Embedding(7, embedding_dim)
        self.day_of_year_embedding = nn.Embedding(366, embedding_dim)
        
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
        
        self.final_conv = nn.Sequential(
            ResnetBlock1D(hidden_dims[0], hidden_dims[0], time_emb_dim=time_emb_dim, dropout=dropout), 
            nn.Conv1d(hidden_dims[0], in_channels, 1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, conditions: Dict[str, torch.Tensor], 
                conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop(), emb)
            
        return self.final_conv(x).permute(0, 2, 1)


class ImprovedDiffusionModel(nn.Module):
    def __init__(self, base_model: ConditionalUnet, num_timesteps: int):
        super().__init__()
        self.model = base_model
        self.num_timesteps = num_timesteps
        
        betas = self._cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x_0: torch.Tensor, conditions: Dict[str, torch.Tensor], 
                conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t, conditions, conditioning_signal)
        return F.huber_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, num_samples: int, conditions: Dict[str, torch.Tensor], shape: tuple, 
               conditioning_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.randn(num_samples, *shape, device=device)
        
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps, leave=False):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch, conditions, conditioning_signal)
            
            alpha_t = self.alphas[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
                
        return x


class HierarchicalDiffusionModel(nn.Module):
    def __init__(self, in_channels: int, num_houses: int, downscale_factor: int, **model_kwargs):
        super().__init__()
        self.downscale_factor = downscale_factor
        self.fine_chunk_size = 2 * 96

        num_timesteps = model_kwargs.pop("num_timesteps")
        
        self.downsampler = nn.Conv1d(in_channels, in_channels, kernel_size=downscale_factor, stride=downscale_factor)
        self.upsampler = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=downscale_factor, stride=downscale_factor)
        
        self.coarse_model = ImprovedDiffusionModel(
            ConditionalUnet(in_channels=in_channels, num_houses=num_houses, **model_kwargs), 
            num_timesteps
        )
        self.fine_model = ImprovedDiffusionModel(
            ConditionalUnet(in_channels=in_channels, num_houses=num_houses, 
                          cond_channels=in_channels, **model_kwargs), 
            num_timesteps
        )

    def forward(self, x_0: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_0_coarse = self.downsampler(x_0.permute(0, 2, 1)).permute(0, 2, 1)
        coarse_loss = self.coarse_model(x_0_coarse, conditions)
        
        with torch.no_grad():
            x_0_coarse_upsampled = self.upsampler(x_0_coarse.detach().permute(0, 2, 1)).permute(0, 2, 1)
            
            if x_0_coarse_upsampled.shape[1] != x_0.shape[1]:
                diff = x_0.shape[1] - x_0_coarse_upsampled.shape[1]
                if diff > 0: x_0_coarse_upsampled = F.pad(x_0_coarse_upsampled, [0, 0, 0, diff])
                else: x_0_coarse_upsampled = x_0_coarse_upsampled[:, :x_0.shape[1], :]
            x_0_fine_residual = x_0 - x_0_coarse_upsampled
        
        full_length = x_0.shape[1]
        if full_length > self.fine_chunk_size:
            start_index = torch.randint(0, full_length - self.fine_chunk_size + 1, (1,)).item()
        else:
            start_index = 0
            self.fine_chunk_size = full_length
        
        residual_chunk = x_0_fine_residual[:, start_index:start_index + self.fine_chunk_size, :]
        conditioning_chunk = x_0_coarse_upsampled[:, start_index:start_index + self.fine_chunk_size, :]
        
        fine_loss = self.fine_model(residual_chunk, conditions, conditioning_signal=conditioning_chunk)
        
        fine_loss_weight = 1.5
        return coarse_loss + (fine_loss * fine_loss_weight)

    @torch.no_grad()
    def sample(self, num_samples: int, conditions: Dict[str, torch.Tensor], shape: tuple) -> torch.Tensor:
        full_length, num_features = shape
        device = next(self.parameters()).device
        
        conditions = {k: v.to(device) for k, v in conditions.items()}
        
        print("--- Stage 1: Sampling Coarse Structure ---")
        coarse_shape = (full_length // self.downscale_factor, num_features)
        generated_coarse = self.coarse_model.sample(num_samples, conditions, shape=coarse_shape)
        upsampled_coarse = self.upsampler(generated_coarse.permute(0, 2, 1)).permute(0, 2, 1)
        
        if upsampled_coarse.shape[1] != full_length:
            diff = full_length - upsampled_coarse.shape[1]
            if diff > 0: upsampled_coarse = F.pad(upsampled_coarse, [0, 0, 0, diff])
            else: upsampled_coarse = upsampled_coarse[:, :full_length, :]
        
        print("--- Stage 2: Sampling Fine Details ---")
        stitched_fine_residual = torch.zeros_like(upsampled_coarse)
        
        for start_index in tqdm(range(0, full_length, self.fine_chunk_size), desc="Fine chunks"):
            end_index = min(start_index + self.fine_chunk_size, full_length)
            chunk_length = end_index - start_index
            fine_shape = (chunk_length, num_features)
            conditioning_chunk = upsampled_coarse[:, start_index:end_index, :]
            
            generated_fine_chunk = self.fine_model.sample(
                num_samples, conditions, shape=fine_shape, 
                conditioning_signal=conditioning_chunk
            )
            
            stitched_fine_residual[:, start_index:end_index, :] = generated_fine_chunk
        
        final_sample = upsampled_coarse + stitched_fine_residual
        return final_sample
