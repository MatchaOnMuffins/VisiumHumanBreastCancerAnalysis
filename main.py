#!/usr/bin/env python3

import argparse

import scanpy as sc
import squidpy
import torch
from sctypepy import run_sctype
from sklearn.preprocessing import StandardScaler
from src.config.config import load_config
from src.dataloader.data import (
    build_spatial_graph,
    load_visium_data,
    preprocess_expression,
    create_spatial_features,
)
from src.model.model import SpatialVAE, ConditionalSpatialVAE
from src.training.train import train_model, train_model_conditional
import pandas as pd

custom_db = pd.DataFrame(
        {
            "tissueType": ["Breast"] * 10,
            "cellName": [
                "Smooth muscle",
                "Endothelial",
                "Luminal epithelial",
                "Stroma fibroblasts",
                "Endothelial subset",
                "Mesenchymal progenitor",
                "Stroma fibroblast subset",
                "Basal/myoepithelial",
                "Immune CTL",
                "Other epithelial",
            ],
            "geneSymbolmore1": [
                "ACTA2, MYH11, MYL9, MYLK, TAGLN",
                "EMCN, ENG, PLVAP, SELE, SELP",
                "KRT18, KRT19, KRT7, KRT8",
                "COL1A1, COL1A2, COL3A1, COL6A1, COL6A2, COL6A3, ALDH1A1",
                "EDN1, EFNB2, ELTD1, ESAM",
                "APOE, CFD, STEAP4, CCL2, CEBPD, COL4A1, COL6A3, SNAI2, TGFB1",
                "COL14A1, COL1A1, COL1A2, COL6A2, ALDH1A1",
                "KRT5, KRT14, KRT17, ACTA2, TAGLN",
                "C2, CD3D, CD7, CD8A, CD69, IL1B, TNF, GZMK",
                "EPCAM, KRT6B, KRT15, KRT16, KRT81, KRT23",
            ],
            "geneSymbolmore2": [""] * 10,
        }
    )


def main_svae(config_path: str = "config.yaml"):
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # print("\n1. Loading data...")
    adata = load_visium_data(config.data_dir, config.counts_file)
    expression_matrix = adata.X.toarray()  # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(
        f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes"
    )

    # === Approach 1: Standard PCA-based clustering ===
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, key_added="leiden_pca")
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_pca", db=custom_db)
    adata.obs["sctype_pca"] = adata.obs["sctype_classification"]

    # === Approach 2: Spatial VAE-based clustering ===

    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=config.min_spots_per_gene,
        n_top_genes=config.top_genes_count,
        n_pca_components=config.pca_components,
    )
    edge_index = build_spatial_graph(spatial_coordinates, k=config.spatial_neighbors)

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


    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(
        adata, use_rep="X_spatial_vae", n_neighbors=config.cluster_neighbors
    )
    sc.tl.leiden(adata, key_added="leiden_svae")


    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_svae", db=custom_db)
    adata.obs["sctype_svae"] = adata.obs["sctype_classification"]
    adata.obs[["leiden_pca", "leiden_svae", "sctype_pca", "sctype_svae"]].to_csv("cluster_assignments.csv")

    sc.tl.umap(adata)

    # PCA-based results
    squidpy.pl.spatial_scatter(adata, color="sctype_pca", title="PCA-based cell types")
    sc.pl.umap(adata, color="sctype_pca", title="UMAP - PCA clustering")

    # Spatial VAE-based results
    squidpy.pl.spatial_scatter(adata, color="sctype_svae", title="Spatial VAE-based cell types")
    sc.pl.umap(adata, color="sctype_svae", title="UMAP - Spatial VAE clustering")

    # Leiden clusters comparison
    squidpy.pl.spatial_scatter(adata, color=["leiden_pca", "leiden_svae"], title=["PCA Leiden", "SVAE Leiden"])
    sc.pl.umap(adata, color=["leiden_pca", "leiden_svae"], title=["PCA Leiden", "SVAE Leiden"])


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


    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata)
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden", db=custom_db)

    spatial_features, names = create_spatial_features(adata, k=6)
    adata.obsm["spatial_features"] = spatial_features
    adata.uns["spatial_feature_names"] = names

    print(f"   Created Spatial features: {names}")
    print(f"   PCA shape: {pca_features.shape}")

    print("\n3. Building spatial k-NN graph...")
    edge_index = build_spatial_graph(spatial_coordinates, k=config.spatial_neighbors)
    print(f"   Graph edges: {edge_index.shape[1]}")

    print("\n4. Training Conditional Spatial VAE...")
    model = ConditionalSpatialVAE(
        x_dim=pca_features.shape[1],
        s_dim=spatial_features.shape[1],
        latent_dim=config.latent_dimensions,
        hidden_dims=config.hidden_layer_dimensions,
    )
    embeddings = train_model_conditional(
        model,
        pca_features,
        spatial_features,
        edge_index,
        n_epochs=config.training_epochs,
        lr=config.learning_rate,
        lambda_spatial=config.spatial_regularization_weight,
        device=device,
        beta=config.beta,
    )

    print("\n5. Clustering and visualization...")
    adata.obsm["X_cvae"] = StandardScaler().fit_transform(embeddings)

    sc.pp.neighbors(
        adata, use_rep="X_cvae", n_neighbors=config.cluster_neighbors
    )
    sc.tl.leiden(adata)
    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    squidpy.pl.spatial_scatter(adata, color="leiden", title="Spatial clusters")
    sc.pl.umap(adata, color="leiden", title="UMAP of latent space")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_cvae.yaml",
        help="Path to configuration YAML file (default: config_cvae.yaml)",
    )
    args = parser.parse_args()

    # main_cvae(config_path=args.config)
    main_svae(config_path=args.config)