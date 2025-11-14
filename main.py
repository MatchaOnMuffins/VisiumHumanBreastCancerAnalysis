#!/usr/bin/env python3

import torch
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from src.data import load_visium_data, preprocess_expression, build_spatial_graph
from src.model import SpatialVAE
from src.train import train_model

DATA_DIR = "data/"
COUNTS_FILE = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

MIN_SPOTS_PER_GENE = 200
TOP_GENES_COUNT = 3000
PCA_COMPONENTS = 20
SPATIAL_NEIGHBORS = 6

LATENT_DIMENSIONS = 32
HIDDEN_LAYER_DIMENSIONS = (256, 128)
TRAINING_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SPATIAL_REGULARIZATION_WEIGHT = 3.0

CLUSTER_NEIGHBORS = 15


def load_and_prepare_data(data_directory, counts_filename):
    print("\n1. Loading data...")
    adata = load_visium_data(data_directory, counts_filename)
    expression_matrix = adata.X.toarray() # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes")
    return adata, expression_matrix, spatial_coordinates


def create_expression_features(expression_matrix, min_spots, top_genes, pca_components):
    print("\n2. Preprocessing expression data...")
    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=min_spots,
        n_top_genes=top_genes,
        n_pca_components=pca_components
    )
    print(f"   PCA shape: {pca_features.shape}")
    return pca_features


def create_spatial_graph(coordinates, neighbors):
    print("\n3. Building spatial k-NN graph...")
    edge_index = build_spatial_graph(coordinates, k=neighbors)
    print(f"   Graph edges: {edge_index.shape[1]}")
    return edge_index


def train_spatial_vae(pca_features, edge_index, latent_dim, hidden_dims, epochs, batch_size, lr, lambda_spatial, device):
    print("\n4. Training Spatial VAE...")
    model = SpatialVAE(in_dim=pca_features.shape[1], latent_dim=latent_dim, hidden_dims=hidden_dims)
    embeddings = train_model(
        model, pca_features, edge_index,
        n_epochs=epochs, batch_size=batch_size, lr=lr,
        lambda_spatial=lambda_spatial, device=device
    )
    return embeddings


def cluster_and_visualize(adata, embeddings, neighbors):
    print("\n5. Clustering and visualization...")
    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(adata, use_rep="X_spatial_vae", n_neighbors=neighbors)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    sc.pl.spatial(adata, color="leiden", title="Spatial clusters")
    sc.pl.umap(adata, color="leiden", title="UMAP of latent space")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    adata, expression_matrix, spatial_coordinates = load_and_prepare_data(DATA_DIR, COUNTS_FILE)

    pca_features = create_expression_features(
        expression_matrix, MIN_SPOTS_PER_GENE, TOP_GENES_COUNT, PCA_COMPONENTS
    )

    edge_index = create_spatial_graph(spatial_coordinates, SPATIAL_NEIGHBORS)

    embeddings = train_spatial_vae(
        pca_features, edge_index, LATENT_DIMENSIONS, HIDDEN_LAYER_DIMENSIONS,
        TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, SPATIAL_REGULARIZATION_WEIGHT, device
    )

    cluster_and_visualize(adata, embeddings, CLUSTER_NEIGHBORS)

    print("\nDone!")


if __name__ == "__main__":
    main()
