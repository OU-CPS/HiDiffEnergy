import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from tqdm import tqdm

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
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        h = self.block2_act(self.block2_norm(h))
        h = self.block2_dropout(h)
        h = self.block2_conv(h)

        return h + self.res_conv(x)

class AttentionBlock1D(nn.Module):
    """Self-attention block for 1D sequences - CORRECTED."""
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
        
        # Generate Q, K, V
        qkv = self.qkv(h)  # (B, 3*C, L)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, L)  # (B, 3, num_heads, head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, L, head_dim)   
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # (B, num_heads, L, head_dim)
        out = out.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, L)
        out = out.contiguous().view(B, C, L)  # (B, C, L)
        
        return x + self.proj(out)

class DownBlock1D(nn.Module):
    """Downsampling block for the U-Net encoder - CORRECTED."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float, use_attention: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock1D(in_channels, out_channels, time_emb_dim=time_emb_dim, dropout=dropout),
            ResnetBlock1D(out_channels, out_channels, time_emb_dim=time_emb_dim, dropout=dropout),
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
    """Upsampling block for the U-Net decoder."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float, use_attention: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock1D(in_channels * 2, out_channels, time_emb_dim=time_emb_dim, dropout=dropout),
            ResnetBlock1D(out_channels, out_channels, time_emb_dim=time_emb_dim, dropout=dropout),
        ])
        self.attn = AttentionBlock1D(out_channels) if use_attention else nn.Identity()
        self.upsampler = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip_x, time_emb):
        x = self.upsampler(x)
        x = torch.cat([skip_x, x], dim=1)
        for resnet in self.resnets:
            x = resnet(x, time_emb)
        return self.attn(x)

class ConditionalUnet(nn.Module):
    """A 1D Conditional U-Net for time series diffusion."""
    def __init__(
        self,
        in_channels: int,
        num_houses: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [64, 128, 256, 512],
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        time_emb_dim = hidden_dims[0] * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        self.house_proj = nn.Linear(embedding_dim, time_emb_dim)

        self.init_conv = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=7, padding=3)
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_blocks.append(
                DownBlock1D(hidden_dims[i], hidden_dims[i+1], time_emb_dim, dropout, use_attention)
            )

        self.mid_block1 = ResnetBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim=time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock1D(hidden_dims[-1])
        self.mid_block2 = ResnetBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim=time_emb_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.up_blocks.append(
                UpBlock1D(hidden_dims[i+1], hidden_dims[i], time_emb_dim, dropout, use_attention)
            )
        
        self.final_conv = nn.Sequential(
            ResnetBlock1D(hidden_dims[0], hidden_dims[0], time_emb_dim=time_emb_dim, dropout=dropout),
            nn.Conv1d(hidden_dims[0], in_channels, 1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timestep)
        house_emb = self.house_embedding(house_id)
        house_emb_proj = self.house_proj(house_emb)
        
        emb = time_emb + house_emb_proj
        
        x = x.permute(0, 2, 1)
        x = self.init_conv(x)
        
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip_x = down_block(x, emb)
            skip_connections.append(skip_x)
        
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        for up_block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = up_block(x, skip_x, emb)
            
        x = self.final_conv(x)
        return x.permute(0, 2, 1)
class ImprovedDiffusionModel(nn.Module):
    """Wrapper for the U-Net model with diffusion logic - CORRECTED variance."""
    def __init__(self, base_model: ConditionalUnet, num_timesteps: int = 300):
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
        # Clipping the variance as in the original DDPM paper
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x_0: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        predicted_noise = self.model(x_t, t, house_id)
        
        return F.huber_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, num_samples: int, house_ids: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Sampling with improved variance (DDPM formulation)."""
        device = house_ids.device
        x = torch.randn(num_samples, *shape, device=device)
        
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps, leave=False):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch, house_ids)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            x_0_pred = (x - self.sqrt_one_minus_alphas_cumprod[t] * predicted_noise) / self.sqrt_alphas_cumprod[t]
            x_0_pred = torch.clamp(x_0_pred, -1, 1)  

            posterior_mean = (
                self.posterior_mean_coef1[t] * x_0_pred +
                self.posterior_mean_coef2[t] * x
            )
            
            if t > 0:
                noise = torch.randn_like(x)
                posterior_variance = self.posterior_variance[t]
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean
                
        return x
