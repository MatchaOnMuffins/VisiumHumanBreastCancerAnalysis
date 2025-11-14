import torch.nn.functional as F
from typing import Optional
import torch
import logging


def spatial_smoothness_loss(latent_embeddings: torch.Tensor, edge_index: torch.Tensor):
    src, dst = edge_index
    diffs = latent_embeddings[src] - latent_embeddings[dst]
    return diffs.pow(2).sum(dim=1).mean()


def reconstruction_loss(reconstructed: torch.Tensor, original: torch.Tensor):
    return F.mse_loss(reconstructed, original, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()


def spatial_vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    latent_embeddings: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    lambda_spatial: float = 0.0,
    beta: float = 1.0,
):
    if edge_index is None:
        logging.warning("edge_index is None, spatial smoothness loss will be skipped.")

    recon_loss = reconstruction_loss(reconstructed, original)
    kl_div = kl_divergence(mu, logvar)

    spatial = (
        spatial_smoothness_loss(latent_embeddings, edge_index)
        if edge_index is not None
        else original.new_zeros(())
    )

    total = recon_loss + beta * kl_div + lambda_spatial * spatial
    return total, recon_loss, kl_div, spatial
