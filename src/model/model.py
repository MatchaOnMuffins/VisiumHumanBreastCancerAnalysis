import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
        super().__init__()

        layers = []
        current = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(nn.ReLU())
            current = h

        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(current, latent_dim)
        self.logvar = nn.Linear(current, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Tuple[int, ...], out_dim: int):
        super().__init__()

        layers = []
        current = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(current, h))
            layers.append(nn.ReLU())
            current = h

        layers.append(nn.Linear(current, out_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        return self.backbone(z)


class SpatialVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, in_dim)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
