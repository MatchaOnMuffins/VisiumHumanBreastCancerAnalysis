import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialVAE(nn.Module):
    """Variational Autoencoder with spatial smoothness regularization."""

    def __init__(self, in_dim, latent_dim=64, hidden_dims=(512, 256)):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder(in_dim, hidden_dims)
        encoder_output_dim = hidden_dims[-1]

        self.mu_layer = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar_layer = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder = self._build_decoder(latent_dim, hidden_dims, in_dim)

    def _build_encoder(self, input_dim, hidden_dims):
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, hidden_dims, output_dim):
        layers = []
        current_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def encode(self, x):
        encoded = self.encoder(x)
        return self.mu_layer(encoded), self.logvar_layer(encoded)

    def reparameterize(self, mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(standard_deviation)
        return mu + epsilon * standard_deviation

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z


def spatial_smoothness_loss(latent_embeddings, edge_index):
    """Compute mean squared distance between connected nodes in latent space."""
    source_nodes, target_nodes = edge_index
    differences = latent_embeddings[source_nodes] - latent_embeddings[target_nodes]
    return differences.pow(2).sum(dim=1).mean()


def compute_reconstruction_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original, reduction="mean")


def compute_kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_loss(reconstructed, original, mu, logvar, latent_embeddings, edge_index=None, lambda_spatial=0.0):
    """Compute VAE loss with optional spatial smoothness term."""
    reconstruction_loss = compute_reconstruction_loss(reconstructed, original)
    kl_divergence = compute_kl_divergence(mu, logvar)

    spatial_loss = torch.tensor(0.0, device=original.device)
    if edge_index is not None and lambda_spatial > 0:
        spatial_loss = spatial_smoothness_loss(latent_embeddings, edge_index)

    total_loss = reconstruction_loss + kl_divergence + lambda_spatial * spatial_loss
    return total_loss, reconstruction_loss, kl_divergence, spatial_loss
