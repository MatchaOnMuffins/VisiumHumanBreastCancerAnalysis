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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n1. Loading data...")
    adata = load_visium_data(DATA_DIR, COUNTS_FILE)
    expression_matrix = adata.X.toarray() # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes")

    print("\n2. Preprocessing expression data...")
    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=MIN_SPOTS_PER_GENE,
        n_top_genes=TOP_GENES_COUNT,
        n_pca_components=PCA_COMPONENTS
    )
    print(f"   PCA shape: {pca_features.shape}")

    print("\n3. Building spatial k-NN graph...")
    edge_index = build_spatial_graph(spatial_coordinates, k=SPATIAL_NEIGHBORS)
    print(f"   Graph edges: {edge_index.shape[1]}")

    print("\n4. Training Spatial VAE...")
    model = SpatialVAE(in_dim=pca_features.shape[1], latent_dim=LATENT_DIMENSIONS, hidden_dims=HIDDEN_LAYER_DIMENSIONS)
    embeddings = train_model(
        model, pca_features, edge_index,
        n_epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
        lambda_spatial=SPATIAL_REGULARIZATION_WEIGHT, device=device
    )

    print("\n5. Clustering and visualization...")
    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(adata, use_rep="X_spatial_vae", n_neighbors=CLUSTER_NEIGHBORS)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    sc.pl.spatial(adata, color="leiden", title="Spatial clusters")
    sc.pl.umap(adata, color="leiden", title="UMAP of latent space")

    print("\nDone!")


if __name__ == "__main__":
    main()
