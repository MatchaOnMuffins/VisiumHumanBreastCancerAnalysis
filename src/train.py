import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import compute_loss


class TrainingMetrics:
    def __init__(self):
        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_divergence = 0.0
        self.num_samples = 0

    def accumulate(self, total, reconstruction, kl, batch_size):
        self.total_loss += total * batch_size
        self.reconstruction_loss += reconstruction * batch_size
        self.kl_divergence += kl * batch_size
        self.num_samples += batch_size

    def compute_averages(self):
        return (
            self.total_loss / self.num_samples,
            self.reconstruction_loss / self.num_samples,
            self.kl_divergence / self.num_samples
        )


def prepare_data_for_training(data, edge_index, device):
    data_tensor = torch.from_numpy(data).to(device)
    edge_index_tensor = edge_index.to(device)
    return data_tensor, edge_index_tensor


def create_data_loader(data_tensor, batch_size):
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_single_batch(model, batch, optimizer):
    reconstruction, mu, logvar, latent = model(batch)
    total_loss, recon_loss, kl_loss, _ = compute_loss(
        reconstruction, batch, mu, logvar, latent
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), recon_loss.item(), kl_loss.item()


def run_training_epoch(model, data_loader, optimizer):
    model.train()
    metrics = TrainingMetrics()

    for (batch_data,) in data_loader:
        total, reconstruction, kl = train_single_batch(model, batch_data, optimizer)
        metrics.accumulate(total, reconstruction, kl, batch_data.size(0))

    return metrics.compute_averages()


def evaluate_full_dataset(model, full_data, edge_index, lambda_spatial):
    model.eval()
    with torch.no_grad():
        reconstruction, mu, logvar, latent = model(full_data)
        total_loss, recon_loss, kl_loss, spatial_loss = compute_loss(
            reconstruction, full_data, mu, logvar, latent,
            edge_index=edge_index, lambda_spatial=lambda_spatial
        )
    return total_loss, recon_loss, kl_loss, spatial_loss


def print_epoch_metrics(epoch, train_metrics, eval_metrics):
    train_total, train_recon, train_kl = train_metrics
    eval_total, eval_recon, eval_kl, eval_spatial = eval_metrics

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train: {train_total:.4f} (R:{train_recon:.4f} KL:{train_kl:.4f}) | "
        f"Eval: {eval_total:.4f} (R:{eval_recon:.4f} KL:{eval_kl:.4f} S:{eval_spatial:.4f})"
    )


def extract_final_embeddings(model, full_data):
    model.eval()
    with torch.no_grad():
        _, _, _, latent_embeddings = model(full_data)
    return latent_embeddings.cpu().numpy()


def train_model(model, X, edge_index, n_epochs=50, batch_size=256, lr=1e-3, lambda_spatial=5.0, device="cpu"):
    """Train SpatialVAE model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    full_data, edge_index_gpu = prepare_data_for_training(X, edge_index, device)
    data_loader = create_data_loader(full_data, batch_size)

    for epoch in range(n_epochs):
        train_metrics = run_training_epoch(model, data_loader, optimizer)
        eval_metrics = evaluate_full_dataset(model, full_data, edge_index_gpu, lambda_spatial)
        print_epoch_metrics(epoch, train_metrics, eval_metrics)

    return extract_final_embeddings(model, full_data)
