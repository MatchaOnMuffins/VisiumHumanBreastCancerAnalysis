import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import compute_loss


def train_model(model, X, edge_index, n_epochs=50, batch_size=256, lr=1e-3, lambda_spatial=5.0, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    full_data = torch.from_numpy(X).to(device)
    edge_index_gpu = edge_index.to(device)

    dataset = TensorDataset(full_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_samples = 0

        for (batch_data,) in data_loader:
            reconstruction, mu, logvar, latent = model(batch_data)
            total_loss, recon_loss, kl_loss, _ = compute_loss(
                reconstruction, batch_data, mu, logvar, latent
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size_actual = batch_data.size(0)
            total_loss_sum += total_loss.item() * batch_size_actual
            recon_loss_sum += recon_loss.item() * batch_size_actual
            kl_loss_sum += kl_loss.item() * batch_size_actual
            num_samples += batch_size_actual

        train_total = total_loss_sum / num_samples
        train_recon = recon_loss_sum / num_samples
        train_kl = kl_loss_sum / num_samples

        model.eval()
        with torch.no_grad():
            reconstruction, mu, logvar, latent = model(full_data)
            eval_total, eval_recon, eval_kl, eval_spatial = compute_loss(
                reconstruction, full_data, mu, logvar, latent,
                edge_index=edge_index_gpu, lambda_spatial=lambda_spatial
            )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train: {train_total:.4f} (R:{train_recon:.4f} KL:{train_kl:.4f}) | "
            f"Eval: {eval_total.item():.4f} (R:{eval_recon.item():.4f} KL:{eval_kl.item():.4f} S:{eval_spatial.item():.4f})"
        )

    model.eval()
    with torch.no_grad():
        _, _, _, latent_embeddings = model(full_data)

    return latent_embeddings.cpu().numpy()
