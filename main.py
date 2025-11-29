#!/usr/bin/env python3

import argparse
import torch
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from src.dataloader.data import (
    load_visium_data,
    preprocess_expression,
    build_spatial_graph,
    annotate_cell_types, compute_cell_type_fractions,
    create_spatial_features
)
from src.model.model import SpatialVAE, ConditionalSpatialVAE
from src.training.train import train_model
from src.config.config import load_config


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n1. Loading data...")
    adata = load_visium_data(config.data_dir, config.counts_file)
    expression_matrix = adata.X.toarray()  # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(
        f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes"
    )

    print("\n2. Preprocessing expression data...")
    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=config.min_spots_per_gene,
        n_top_genes=config.top_genes_count,
        n_pca_components=config.pca_components,
    )
    print(f"   PCA shape: {pca_features.shape}")

    print("\n3. Building spatial k-NN graph...")
    edge_index = build_spatial_graph(spatial_coordinates, k=config.spatial_neighbors)
    print(f"   Graph edges: {edge_index.shape[1]}")

    print("\n4. Training Spatial VAE...")
    model = SpatialVAE(
        in_dim=pca_features.shape[1],
        latent_dim=config.latent_dimensions,
        hidden_dims=config.hidden_layer_dimensions,
    )
    embeddings = train_model(
        model,
        pca_features,
        edge_index,
        n_epochs=config.training_epochs,
        lr=config.learning_rate,
        lambda_spatial=config.spatial_regularization_weight,
        device=device,
    )

    print("\n5. Clustering and visualization...")
    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(
        adata, use_rep="X_spatial_vae", n_neighbors=config.cluster_neighbors
    )
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    sc.pl.spatial(adata, color="leiden", title="Spatial clusters")
    sc.pl.umap(adata, color="leiden", title="UMAP of latent space")

    print("\nDone!")

def main_cvae(config_path: str = "config.yaml"):
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n1. Loading data...")
    adata = load_visium_data(config.data_dir, config.counts_file)
    expression_matrix = adata.X.toarray()  # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(
        f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes"
    )

    print("\n2. Preprocessing expression data...")
    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=config.min_spots_per_gene,
        n_top_genes=config.top_genes_count,
        n_pca_components=config.pca_components,
    )
    annotate_cell_types(adata, tissue="Kidney")
    spatial_features, names = create_spatial_features(adata, k=6)
    adata.obsm["spatial_features"] = spatial_features
    adata.uns["spatial_feature_names"] = names

    print(f"   Created Spatial features: {names}")
    print(f"   PCA shape: {pca_features.shape}")

    print("\n3. Building spatial k-NN graph...")
    edge_index = build_spatial_graph(spatial_coordinates, k=config.spatial_neighbors)
    print(f"   Graph edges: {edge_index.shape[1]}")

    print("\n4. Training Spatial VAE...")
    model = ConditionalSpatialVAE(
        in_dim=pca_features.shape[1],
        s_dim=spatial_features.shape[1],
        latent_dim=config.latent_dimensions,
        hidden_dims=-config.hidden_layer_dimensions,
    )
    embeddings = train_model(
        model,
        pca_features,
        edge_index,
        n_epochs=config.training_epochs,
        lr=config.learning_rate,
        lambda_spatial=config.spatial_regularization_weight,
        device=device,
    )

    print("\n5. Clustering and visualization...")
    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(
        adata, use_rep="X_spatial_vae", n_neighbors=config.cluster_neighbors
    )
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    sc.pl.spatial(adata, color="leiden", title="Spatial clusters")
    sc.pl.umap(adata, color="leiden", title="UMAP of latent space")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    main(config_path=args.config)
