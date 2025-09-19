# autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ImprovedConditionalVAE(nn.Module):
    """
    A Conditional VAE improved with Dropout in the decoder to help 
    prevent posterior collapse.
    """
    def __init__(self, in_channels: int, num_houses: int, window_size: int, 
                 latent_dim: int = 64, embedding_dim: int = 64, 
                 hidden_dims: list = [32, 64, 128], dropout_rate: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()

        # --- House ID Embedding ---
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        
        # --- Encoder ---
        encoder_modules = []
        current_in_channels = in_channels + embedding_dim
        for h_dim in self.hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv1d(current_in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            current_in_channels = h_dim
        self.encoder_conv = nn.Sequential(*encoder_modules)
        
        self.final_encoder_seq_len = window_size // (2**len(self.hidden_dims))
        
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_log_var = nn.LazyLinear(latent_dim)

        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim + embedding_dim, self.hidden_dims[-1] * self.final_encoder_seq_len)

        decoder_modules = []
        self.hidden_dims.reverse() # Reverse for upsampling 
        
        current_in_channels = self.hidden_dims[0]
        for i, h_dim in enumerate(self.hidden_dims[1:]):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(current_in_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            current_in_channels = h_dim
        self.decoder_conv = nn.Sequential(*decoder_modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(current_in_channels, current_in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(current_in_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate), 
            nn.Conv1d(current_in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor, house_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        house_emb = self.house_embedding(house_id).unsqueeze(2).expand(-1, -1, x.size(2))
        combined_input = torch.cat([x, house_emb], dim=1)
        
        result = self.encoder_conv(combined_input)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        house_emb = self.house_embedding(house_id)
        combined_input = torch.cat([z, house_emb], dim=1)
        
        result = self.decoder_input(combined_input)
        result = result.view(-1, self.hidden_dims[0], self.final_encoder_seq_len)
        
        result = self.decoder_conv(result)
        reconstruction = self.final_layer(result)
        return reconstruction

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, house_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        mu, log_var = self.encode(x, house_id)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, house_id)
        return reconstruction.permute(0, 2, 1), mu, log_var
        
    def sample(self, num_samples: int, house_id: torch.Tensor, device: str) -> torch.Tensor:
        self.eval() # Ensure model is in eval mode for sampling
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z, house_id)
        return samples.permute(0, 2, 1)