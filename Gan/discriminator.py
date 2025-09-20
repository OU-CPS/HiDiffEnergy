import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    
    def __init__(self, num_houses: int, embedding_dim: int, 
                 seq_len: int = 96, num_features: int = 4, hidden_size: int = 128):
        super().__init__()
        self.house_embedding = nn.Sequential(
            nn.Embedding(num_houses, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.1)
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_features, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.2),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(hidden_size * 4),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.Conv1d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(hidden_size * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_size * 8, hidden_size * 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(hidden_size * 8),
            nn.LeakyReLU(0.2), 
        )
        final_conv_features = hidden_size * 8 * (seq_len // 16)
        self.classifier_head = nn.Sequential(
            nn.Linear(final_conv_features + embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, series: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        series_permuted = series.permute(0, 2, 1)
        conv_out = self.conv_layers(series_permuted)
        conv_out_flat = torch.flatten(conv_out, 1)
        house_emb = self.house_embedding(house_id)
        combined = torch.cat([conv_out_flat, house_emb], dim=1)
        logit = self.classifier_head(combined)
        return logit

class SpectralDiscriminator(nn.Module):
  
    def __init__(self, num_houses: int, embedding_dim: int, 
                 seq_len: int = 96, num_features: int = 4, hidden_size: int = 128):
        super().__init__()
        self.house_embedding = nn.Sequential(
            nn.Embedding(num_houses, embedding_dim),
            nn.utils.spectral_norm(nn.Linear(embedding_dim, embedding_dim)),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.1)
        )
        self.conv_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(num_features, hidden_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm1d(hidden_size * 4),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm1d(hidden_size * 8),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(hidden_size * 8, hidden_size * 8, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm1d(hidden_size * 8),
            nn.LeakyReLU(0.2), 
        )
       
        final_conv_len = seq_len // (2**4) 
        final_conv_features = hidden_size * 8 * final_conv_len
        self.classifier_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(final_conv_features + embedding_dim, 512)),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.5),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.5),
            nn.utils.spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, series: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        series_permuted = series.permute(0, 2, 1)
        conv_out = self.conv_layers(series_permuted)
        conv_out_flat = torch.flatten(conv_out, 1)
        house_emb = self.house_embedding(house_id)
        combined = torch.cat([conv_out_flat, house_emb], dim=1)
        logit = self.classifier_head(combined)
        return logit