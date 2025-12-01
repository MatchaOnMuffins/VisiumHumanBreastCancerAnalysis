#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.evaluation import (
    evaluate_embeddings,
    evaluate_sctype_across_resolutions,
    compute_sctype_stability,
    save_evaluation_results,
    run_sctype_dedup,
)
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

    # === Evaluation: Cluster quality ===
    print("\n=== Evaluating cluster quality ===")
    metrics_df = evaluate_embeddings(adata, spatial_k=config.spatial_neighbors)
    print(metrics_df.to_string(index=False))

    # === sctype robustness across clustering resolutions ===
    resolutions = [0.4, 0.6, 0.8, 1.0, 1.2]

    print("\n=== sctype robustness across resolutions (PCA) ===")
    pca_res_df, pca_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_pca",
        method_prefix="pca",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(pca_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (PCA): {pca_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    print("\n=== sctype robustness across resolutions (sVAE) ===")
    svae_res_df, svae_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_spatial_vae",
        method_prefix="svae",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(svae_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (sVAE): {svae_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    # Save all results
    save_evaluation_results(
        metrics_df,
        sctype_stability=svae_res_summary["mean_sctype_stability_across_resolutions"],
        output_prefix="cluster_metrics_svae",
        output_dir="results",
    )
    pca_res_df.to_csv("results/sctype_robustness_pca.csv", index=False)
    svae_res_df.to_csv("results/sctype_robustness_svae.csv", index=False)

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


    # === Approach 1: Standard PCA-based clustering (for comparison) ===
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, key_added="leiden_pca")
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_pca", db=custom_db)
    adata.obs["sctype_pca"] = adata.obs["sctype_classification"]

    # Set up expr_cluster_id for spatial feature construction
    adata.obs["expr_cluster"] = adata.obs["leiden_pca"].astype("category")
    adata.obs["expr_cluster_id"] = adata.obs["expr_cluster"].cat.codes

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
    sc.tl.leiden(adata, key_added="leiden_cvae")

    # Run sctype on cVAE clusters
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_cvae", db=custom_db)
    adata.obs["sctype_cvae"] = adata.obs["sctype_classification"]

    # Save cluster assignments
    adata.obs[["leiden_pca", "leiden_cvae", "sctype_pca", "sctype_cvae"]].to_csv(
        "results/cluster_assignments_cvae.csv"
    )

    # === Evaluation: Cluster quality ===
    print("\n=== Evaluating cluster quality ===")
    metrics_df = evaluate_embeddings(adata, spatial_k=config.spatial_neighbors)
    print(metrics_df.to_string(index=False))

    # === sctype robustness across clustering resolutions ===
    resolutions = [0.4, 0.6, 0.8, 1.0, 1.2]

    print("\n=== sctype robustness across resolutions (PCA) ===")
    pca_res_df, pca_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_pca",
        method_prefix="pca",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(pca_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (PCA): {pca_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    print("\n=== sctype robustness across resolutions (cVAE) ===")
    cvae_res_df, cvae_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_cvae",
        method_prefix="cvae",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(cvae_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (cVAE): {cvae_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    # Save all results
    save_evaluation_results(
        metrics_df,
        sctype_stability=cvae_res_summary["mean_sctype_stability_across_resolutions"],
        output_prefix="cluster_metrics_cvae",
        output_dir="results",
    )
    pca_res_df.to_csv("results/sctype_robustness_pca_cvae.csv", index=False)
    cvae_res_df.to_csv("results/sctype_robustness_cvae.csv", index=False)

    sc.tl.umap(adata)

    print("\n6. Generating plots...")
    squidpy.pl.spatial_scatter(adata, color="leiden_cvae", title="Spatial clusters (cVAE)")
    sc.pl.umap(adata, color="leiden_cvae", title="UMAP of latent space (cVAE)")

    # sctype-based visualization
    squidpy.pl.spatial_scatter(adata, color="sctype_cvae", title="cVAE-based cell types")
    sc.pl.umap(adata, color="sctype_cvae", title="UMAP - cVAE sctype")

    print("\nDone!")


def run_full_pipeline(config_path: str = "config_cvae.yaml") -> None:
    """
    Run the complete spatial transcriptomics pipeline with all three models.

    This function:
    1. Loads Visium data once
    2. Runs PCA baseline clustering + sctype
    3. Trains SpatialVAE and clusters + sctype
    4. Trains ConditionalSpatialVAE and clusters + sctype
    5. Evaluates all three methods with cluster quality metrics
    6. Computes sctype stability across resolutions for each method
    7. Generates comparative spatial plots

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # 1. Load data (once for all methods)
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Loading Visium data...")
    print("=" * 60)

    adata = load_visium_data(config.data_dir, config.counts_file)
    expression_matrix = adata.X.toarray()  # type: ignore
    spatial_coordinates = adata.obsm["spatial"].astype("float32")
    print(f"   Loaded {expression_matrix.shape[0]} spots, {expression_matrix.shape[1]} genes")

    # =========================================================================
    # 2. PCA Baseline
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Running PCA baseline...")
    print("=" * 60)

    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, key_added="leiden_pca")
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_pca", db=custom_db)
    adata.obs["sctype_pca"] = adata.obs["sctype_classification"]
    print(f"   PCA clusters: {adata.obs['leiden_pca'].nunique()}")
    print(f"   Cell types identified: {adata.obs['sctype_pca'].nunique()}")

    # =========================================================================
    # 3. Spatial VAE
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Training Spatial VAE...")
    print("=" * 60)

    pca_features = preprocess_expression(
        expression_matrix,
        min_spots=config.min_spots_per_gene,
        n_top_genes=config.top_genes_count,
        n_pca_components=config.pca_components,
    )
    edge_index = build_spatial_graph(spatial_coordinates, k=config.spatial_neighbors)

    svae_model = SpatialVAE(
        in_dim=pca_features.shape[1],
        latent_dim=config.latent_dimensions,
        hidden_dims=config.hidden_layer_dimensions,
    )
    svae_embeddings = train_model(
        svae_model,
        pca_features,
        edge_index,
        n_epochs=config.training_epochs,
        lr=config.learning_rate,
        lambda_spatial=config.spatial_regularization_weight,
        device=device,
    )

    adata.obsm["X_spatial_vae"] = StandardScaler().fit_transform(svae_embeddings)

    sc.pp.neighbors(adata, use_rep="X_spatial_vae", n_neighbors=config.cluster_neighbors)
    sc.tl.leiden(adata, key_added="leiden_svae")
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_svae", db=custom_db)
    adata.obs["sctype_svae"] = adata.obs["sctype_classification"]
    print(f"   sVAE clusters: {adata.obs['leiden_svae'].nunique()}")
    print(f"   Cell types identified: {adata.obs['sctype_svae'].nunique()}")

    # =========================================================================
    # 4. Conditional Spatial VAE
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Training Conditional Spatial VAE...")
    print("=" * 60)

    # Create spatial features (uses leiden clustering for neighbor composition)
    # Set up expr_cluster_id from existing PCA leiden clusters
    adata.obs["expr_cluster"] = adata.obs["leiden_pca"].astype("category")
    adata.obs["expr_cluster_id"] = adata.obs["expr_cluster"].cat.codes

    spatial_features, feature_names = create_spatial_features(adata, k=6)
    adata.obsm["spatial_features"] = spatial_features
    adata.uns["spatial_feature_names"] = feature_names
    print(f"   Spatial features: {feature_names}")

    cvae_model = ConditionalSpatialVAE(
        x_dim=pca_features.shape[1],
        s_dim=spatial_features.shape[1],
        latent_dim=config.latent_dimensions,
        hidden_dims=config.hidden_layer_dimensions,
    )
    cvae_embeddings = train_model_conditional(
        cvae_model,
        pca_features,
        spatial_features,
        edge_index,
        n_epochs=config.training_epochs,
        lr=config.learning_rate,
        lambda_spatial=config.spatial_regularization_weight,
        device=device,
        beta=config.beta,
    )

    adata.obsm["X_cvae"] = StandardScaler().fit_transform(cvae_embeddings)

    sc.pp.neighbors(adata, use_rep="X_cvae", n_neighbors=config.cluster_neighbors)
    sc.tl.leiden(adata, key_added="leiden_cvae")
    adata = run_sctype(adata, tissue_type="Breast", groupby="leiden_cvae", db=custom_db)
    adata.obs["sctype_cvae"] = adata.obs["sctype_classification"]
    print(f"   cVAE clusters: {adata.obs['leiden_cvae'].nunique()}")
    print(f"   Cell types identified: {adata.obs['sctype_cvae'].nunique()}")

    # =========================================================================
    # 5. Evaluation: Cluster quality metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. Evaluating cluster quality...")
    print("=" * 60)

    metrics_df = evaluate_embeddings(adata, spatial_k=config.spatial_neighbors)
    print(metrics_df.to_string(index=False))

    # =========================================================================
    # 6. sctype stability across embeddings
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. Computing sctype stability across embeddings...")
    print("=" * 60)

    per_spot_stability, mean_stability = compute_sctype_stability(
        adata, sctype_keys=["sctype_pca", "sctype_svae", "sctype_cvae"]
    )
    print(f"   Mean sctype stability (PCA vs sVAE vs cVAE): {mean_stability:.3f}")

    # =========================================================================
    # 7. sctype robustness across clustering resolutions
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. Evaluating sctype robustness across resolutions...")
    print("=" * 60)

    resolutions = [0.4, 0.6, 0.8, 1.0, 1.2]

    print("\n--- PCA ---")
    pca_res_df, pca_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_pca",
        method_prefix="pca",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(pca_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (PCA): {pca_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    print("\n--- sVAE ---")
    svae_res_df, svae_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_spatial_vae",
        method_prefix="svae",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(svae_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (sVAE): {svae_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    print("\n--- cVAE ---")
    cvae_res_df, cvae_res_summary = evaluate_sctype_across_resolutions(
        adata,
        embedding_key="X_cvae",
        method_prefix="cvae",
        resolutions=resolutions,
        run_sctype_fn=run_sctype,
        tissue_type="Breast",
        db=custom_db,
    )
    print(cvae_res_df[["resolution", "n_clusters", "silhouette"]].to_string(index=False))
    print(f"Mean sctype stability (cVAE): {cvae_res_summary['mean_sctype_stability_across_resolutions']:.3f}")

    # =========================================================================
    # 8. Save all results
    # =========================================================================
    print("\n" + "=" * 60)
    print("8. Saving results...")
    print("=" * 60)

    # Create output directories
    results_dir = Path("results")
    plots_dir = Path("plots")
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # --- Cluster assignments with spot barcodes ---
    cluster_df = adata.obs[[
        "leiden_pca", "leiden_svae", "leiden_cvae",
        "sctype_pca", "sctype_svae", "sctype_cvae"
    ]].copy()
    cluster_df.index.name = "barcode"
    cluster_df.to_csv(results_dir / "cluster_assignments_all.csv")

    # --- Cluster quality metrics (formatted nicely) ---
    metrics_df.to_csv(results_dir / "cluster_quality_metrics.csv", index=False)

    # --- Summary statistics CSV ---
    summary_data = {
        "Method": ["PCA", "sVAE", "cVAE"],
        "Silhouette": [
            metrics_df[metrics_df["method"] == "PCA"]["silhouette"].values[0],
            metrics_df[metrics_df["method"] == "sVAE"]["silhouette"].values[0],
            metrics_df[metrics_df["method"] == "cVAE"]["silhouette"].values[0],
        ],
        "Calinski-Harabasz": [
            metrics_df[metrics_df["method"] == "PCA"]["calinski_harabasz"].values[0],
            metrics_df[metrics_df["method"] == "sVAE"]["calinski_harabasz"].values[0],
            metrics_df[metrics_df["method"] == "cVAE"]["calinski_harabasz"].values[0],
        ],
        "Davies-Bouldin": [
            metrics_df[metrics_df["method"] == "PCA"]["davies_bouldin"].values[0],
            metrics_df[metrics_df["method"] == "sVAE"]["davies_bouldin"].values[0],
            metrics_df[metrics_df["method"] == "cVAE"]["davies_bouldin"].values[0],
        ],
        "Neighborhood_Purity": [
            metrics_df[metrics_df["method"] == "PCA"]["neighborhood_purity"].values[0],
            metrics_df[metrics_df["method"] == "sVAE"]["neighborhood_purity"].values[0],
            metrics_df[metrics_df["method"] == "cVAE"]["neighborhood_purity"].values[0],
        ],
        "Num_Clusters": [
            adata.obs["leiden_pca"].nunique(),
            adata.obs["leiden_svae"].nunique(),
            adata.obs["leiden_cvae"].nunique(),
        ],
        "Num_Cell_Types": [
            adata.obs["sctype_pca"].nunique(),
            adata.obs["sctype_svae"].nunique(),
            adata.obs["sctype_cvae"].nunique(),
        ],
        "sctype_Stability_Across_Resolutions": [
            pca_res_summary["mean_sctype_stability_across_resolutions"],
            svae_res_summary["mean_sctype_stability_across_resolutions"],
            cvae_res_summary["mean_sctype_stability_across_resolutions"],
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary_statistics.csv", index=False)

    # --- Per-spot stability ---
    stability_df = pd.DataFrame({
        "barcode": adata.obs_names,
        "sctype_pca": adata.obs["sctype_pca"].values,
        "sctype_svae": adata.obs["sctype_svae"].values,
        "sctype_cvae": adata.obs["sctype_cvae"].values,
        "stability_score": per_spot_stability.values,
    })
    stability_df.to_csv(results_dir / "sctype_stability_per_spot.csv", index=False)

    # --- Resolution sweep results ---
    pca_res_df.to_csv(results_dir / "resolution_sweep_pca.csv", index=False)
    svae_res_df.to_csv(results_dir / "resolution_sweep_svae.csv", index=False)
    cvae_res_df.to_csv(results_dir / "resolution_sweep_cvae.csv", index=False)

    # --- Combined resolution sweep ---
    pca_res_df["method"] = "PCA"
    svae_res_df["method"] = "sVAE"
    cvae_res_df["method"] = "cVAE"
    combined_res_df = pd.concat([pca_res_df, svae_res_df, cvae_res_df], ignore_index=True)
    combined_res_df.to_csv(results_dir / "resolution_sweep_all.csv", index=False)

    # --- Cell type counts per method ---
    celltype_counts = []
    for method, col in [("PCA", "sctype_pca"), ("sVAE", "sctype_svae"), ("cVAE", "sctype_cvae")]:
        counts = adata.obs[col].value_counts()
        for celltype, count in counts.items():
            celltype_counts.append({"Method": method, "Cell_Type": celltype, "Count": count})
    celltype_df = pd.DataFrame(celltype_counts)
    celltype_df.to_csv(results_dir / "cell_type_counts.csv", index=False)

    print(f"   Saved: {results_dir}/cluster_assignments_all.csv")
    print(f"   Saved: {results_dir}/cluster_quality_metrics.csv")
    print(f"   Saved: {results_dir}/summary_statistics.csv")
    print(f"   Saved: {results_dir}/sctype_stability_per_spot.csv")
    print(f"   Saved: {results_dir}/resolution_sweep_*.csv")
    print(f"   Saved: {results_dir}/cell_type_counts.csv")

    # =========================================================================
    # 9. Generate and save plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("9. Generating and saving plots...")
    print("=" * 60)

    # Configure scanpy to save figures
    sc.settings.figdir = plots_dir
    sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 8))

    # Use non-interactive backend to avoid display issues
    import matplotlib
    matplotlib.use('Agg')

    # Compute UMAP once
    sc.tl.umap(adata)

    # --- Spatial scatter plots using scanpy (more reliable for saving) ---
    # Use sc.pl.spatial which handles saving better
    for col, name in [
        ("sctype_pca", "sctype_pca"),
        ("sctype_svae", "sctype_svae"),
        ("sctype_cvae", "sctype_cvae"),
        ("leiden_pca", "leiden_pca"),
        ("leiden_svae", "leiden_svae"),
        ("leiden_cvae", "leiden_cvae"),
    ]:
        sc.pl.spatial(adata, color=col, save=f"_{name}.png", show=False)

    # --- Comparison plots ---
    sc.pl.spatial(
        adata, color=["sctype_pca", "sctype_svae", "sctype_cvae"],
        save="_sctype_comparison.png", show=False
    )
    sc.pl.spatial(
        adata, color=["leiden_pca", "leiden_svae", "leiden_cvae"],
        save="_leiden_comparison.png", show=False
    )

    # --- UMAP plots for sctype ---
    sc.pl.umap(
        adata,
        color=["sctype_pca", "sctype_svae", "sctype_cvae"],
        title=["PCA", "sVAE", "cVAE"],
        save="_umap_sctype_comparison.png",
        show=False,
    )

    # --- UMAP plots for Leiden clusters ---
    sc.pl.umap(
        adata,
        color=["leiden_pca", "leiden_svae", "leiden_cvae"],
        title=["PCA Leiden", "sVAE Leiden", "cVAE Leiden"],
        save="_umap_leiden_comparison.png",
        show=False,
    )

    # --- Individual UMAP plots ---
    for method, sctype_col, leiden_col in [
        ("pca", "sctype_pca", "leiden_pca"),
        ("svae", "sctype_svae", "leiden_svae"),
        ("cvae", "sctype_cvae", "leiden_cvae"),
    ]:
        sc.pl.umap(adata, color=sctype_col, save=f"_umap_sctype_{method}.png", show=False)
        sc.pl.umap(adata, color=leiden_col, save=f"_umap_leiden_{method}.png", show=False)

    # --- Summary bar chart ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Silhouette scores
    axes[0, 0].bar(summary_df["Method"], summary_df["Silhouette"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0, 0].set_ylabel("Silhouette Score")
    axes[0, 0].set_title("Silhouette Score")
    axes[0, 0].set_ylim(0, max(summary_df["Silhouette"]) * 1.2)

    # Neighborhood purity
    axes[0, 1].bar(summary_df["Method"], summary_df["Neighborhood_Purity"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0, 1].set_ylabel("Neighborhood Purity")
    axes[0, 1].set_title("Spatial Neighborhood Purity")
    axes[0, 1].set_ylim(0, 1)

    # sctype stability
    axes[1, 0].bar(summary_df["Method"], summary_df["sctype_Stability_Across_Resolutions"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 0].set_ylabel("Stability Score")
    axes[1, 0].set_title("sctype Stability Across Resolutions")
    axes[1, 0].set_ylim(0, 1)

    # Number of clusters
    axes[1, 1].bar(summary_df["Method"], summary_df["Num_Clusters"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 1].set_ylabel("Number of Clusters")
    axes[1, 1].set_title("Number of Leiden Clusters")

    plt.tight_layout()
    fig.savefig(plots_dir / "metrics_summary_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   Saved spatial plots to: {plots_dir}/")
    print(f"   Saved UMAP plots to: {plots_dir}/")
    print(f"   Saved: {plots_dir}/metrics_summary_barplot.png")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir.absolute()}")
    print(f"Plots saved to: {plots_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial transcriptomics analysis with VAE models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["svae", "cvae", "all"],
        default="all",
        help="Which pipeline to run: svae, cvae, or all (default: all)",
    )
    args = parser.parse_args()

    if args.mode == "svae":
        main_svae(config_path=args.config)
    elif args.mode == "cvae":
        main_cvae(config_path=args.config)
    else:
        run_full_pipeline(config_path=args.config)