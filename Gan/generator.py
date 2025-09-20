import torch
import torch.nn as nn

class Generator(nn.Module):
    """Base Generator (Corrected: No inplace operations)"""
    def __init__(self, num_houses: int, latent_dim: int, embedding_dim: int, 
                 seq_len: int = 96, num_features: int = 4, hidden_size: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.combined_dim = latent_dim + embedding_dim
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        initial_size = seq_len // (2**4)
        
        self.model = nn.Sequential(
            nn.ConvTranspose1d(self.combined_dim, hidden_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(), 
            
            nn.ConvTranspose1d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_size, self.num_features, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.initial_projection = nn.Linear(self.combined_dim, self.combined_dim * initial_size)
        self.initial_size = initial_size

    def forward(self, z: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        house_emb = self.house_embedding(house_id)
        combined_input = torch.cat([z, house_emb], dim=1)
        x = self.initial_projection(combined_input)
        x = x.view(batch_size, self.combined_dim, self.initial_size)
        output = self.model(x)
        return output.permute(0, 2, 1)

class SpectralGenerator(nn.Module):
    
    def __init__(self, num_houses: int, latent_dim: int, embedding_dim: int, 
                 seq_len: int = 96, num_features: int = 4, hidden_size: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.combined_dim = latent_dim + embedding_dim
        self.house_embedding = nn.Embedding(num_houses, embedding_dim)
        initial_size = seq_len // (2**4)
        
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose1d(self.combined_dim, hidden_size * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_size * 4),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.ConvTranspose1d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2), 
            nn.utils.spectral_norm(nn.ConvTranspose1d(hidden_size, self.num_features, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )
        
        self.initial_projection = nn.utils.spectral_norm(nn.Linear(self.combined_dim, self.combined_dim * initial_size))
        self.initial_size = initial_size

    def forward(self, z: torch.Tensor, house_id: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        house_emb = self.house_embedding(house_id)
        combined_input = torch.cat([z, house_emb], dim=1)
        x = self.initial_projection(combined_input)
        x = x.view(batch_size, self.combined_dim, self.initial_size)
        output = self.model(x)
        return output.permute(0, 2, 1)