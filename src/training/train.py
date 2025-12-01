import torch
from ..loss.loss import spatial_vae_loss, vae_loss


def train_model(
    model,
    X,
    edge_index,
    n_epochs=50,
    lr=1e-3,
    lambda_spatial=5.0,
    device="cpu",
    beta=1.0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.from_numpy(X).to(device)
    edge_index = edge_index.to(device)

    for epoch in range(n_epochs):
        model.train()

        reconstruction, mu, logvar, latent = model(data)
        total_loss, recon_loss, kl_loss, spatial_loss = spatial_vae_loss(
            reconstruction,
            data,
            mu,
            logvar,
            latent,
            edge_index=edge_index,
            lambda_spatial=lambda_spatial,
            beta=beta,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {total_loss.item():.4f} (R:{recon_loss.item():.4f} KL:{kl_loss.item():.4f})"
        )

    model.eval()
    with torch.no_grad():
        _, _, _, latent_embeddings = model(data)

    return latent_embeddings.cpu().numpy()


def train_model_conditional(
    model,
    X,
    S,
    edge_index,
    n_epochs=50,
    lr=1e-3,
    lambda_spatial=5.0,
    device="cpu",
    beta=1.0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data_x = torch.from_numpy(X).to(device).float()
    data_s = torch.from_numpy(S).to(device).float()
    edge_index = edge_index.to(device)

    for epoch in range(n_epochs):
        model.train()

        reconstruction, mu, logvar, latent = model(data_x, data_s)
        total_loss, recon_loss, kl_loss, spatial_loss = spatial_vae_loss(
            reconstruction,
            data_x,
            mu,
            logvar,
            latent,
            edge_index=edge_index,
            lambda_spatial=lambda_spatial,
            beta=beta,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {total_loss.item():.4f} (R:{recon_loss.item():.4f} KL:{kl_loss.item():.4f} S:{spatial_loss.item():.4f})"
        )

    model.eval()
    with torch.no_grad():
        _, _, _, latent_embeddings = model(data_x, data_s)

    return latent_embeddings.cpu().numpy()


def train_model_nospatial(
    model,
    X,
    n_epochs=50,
    lr=1e-3,
    device="cpu",
    beta=1.0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.from_numpy(X).to(device)
    #edge_index = edge_index.to(device)

    for epoch in range(n_epochs):
        model.train()

        reconstruction, mu, logvar, latent = model(data)
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            data,
            mu,
            logvar,
            beta=beta
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {total_loss.item():.4f} (R:{recon_loss.item():.4f} KL:{kl_loss.item():.4f} S:{spatial_loss.item():.4f})"
        )

    model.eval()
    with torch.no_grad():
        _, _, _, latent_embeddings = model(data)

    return latent_embeddings.cpu().numpy()

