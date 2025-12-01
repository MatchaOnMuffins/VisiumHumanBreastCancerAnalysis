"""
Evaluation module for cluster quality and sctype robustness metrics.

This module provides functions to evaluate:
- Internal cluster quality metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
- Spatial coherence via neighborhood purity
- sctype annotation stability across embeddings and clustering resolutions
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sctypepy import run_sctype as _run_sctype_raw


def collapse_duplicate_genes_for_sctype(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Collapse duplicate gene names by summing expression per gene symbol.

    This is used for ScType scoring, which is marker-based and assumes unique
    gene symbols. Instead of using var_names_make_unique() (which renames genes
    and breaks marker matching), we sum expression across duplicated genes.

    Args:
        adata: AnnData with possibly duplicated var_names (gene symbols).

    Returns:
        A new AnnData with unique var_names where duplicate genes have been
        collapsed by summing their expression per spot.
    """
    # Convert to dense if sparse
    if sparse.issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = np.asarray(adata.X)

    # Build DataFrame with obs_names as rows, var_names as columns
    X_df = pd.DataFrame(X_dense, index=adata.obs_names, columns=adata.var_names)

    # Group columns by gene name and sum expression across duplicates
    X_grouped = X_df.groupby(axis=1, level=0).sum()

    # Construct a new AnnData with unique gene names
    adata_unique = anndata.AnnData(
        X_grouped.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=X_grouped.columns),
    )

    # Copy obsm (to preserve spatial/embedding info)
    adata_unique.obsm = adata.obsm.copy()

    return adata_unique


def run_sctype_dedup(
    adata: anndata.AnnData,
    tissue_type: str,
    groupby: str,
    db: Optional[pd.DataFrame] = None,
) -> anndata.AnnData:
    """
    Wrapper around sctypepy.run_sctype that handles duplicate gene names.

    This function:
    1. Collapses duplicate genes for ScType scoring (by summing expression).
    2. Runs ScType on the deduplicated AnnData.
    3. Copies sctype_classification back to the original adata.

    Args:
        adata: AnnData object with expression data and cluster labels.
        tissue_type: Tissue type for ScType database lookup.
        groupby: Column in adata.obs containing cluster labels.
        db: Optional custom marker database DataFrame.

    Returns:
        The original adata with adata.obs["sctype_classification"] filled.
    """
    # 1. Collapse duplicate gene names by summing expression per gene symbol
    adata_sctype = collapse_duplicate_genes_for_sctype(adata)

    # 2. Run ScType on the deduplicated object
    adata_sctype = _run_sctype_raw(
        adata_sctype,
        tissue_type=tissue_type,
        groupby=groupby,
        db=db,
    )

    # 3. Copy sctype_classification back into the original adata
    # Ensure the index aligns
    adata.obs["sctype_classification"] = (
        adata_sctype.obs["sctype_classification"]
        .reindex(adata.obs_names)
        .values
    )

    return adata


def compute_internal_cluster_metrics(
    embedding: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """
    Compute internal cluster quality metrics (no ground truth needed).

    Args:
        embedding: 2D array of shape (n_samples, n_features) - the latent/PCA embedding.
        labels: 1D array of cluster labels for each sample.

    Returns:
        Dictionary with keys: "silhouette", "calinski_harabasz", "davies_bouldin".
        Returns NaN for metrics that cannot be computed (e.g., only 1 cluster).
    """
    # Convert labels to numpy array of integers if categorical
    if hasattr(labels, "cat"):
        labels = labels.cat.codes.values
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Need at least 2 clusters for these metrics
    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    # Need at least 2 samples per cluster for silhouette
    try:
        sil = silhouette_score(embedding, labels)
    except ValueError:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(embedding, labels)
    except ValueError:
        ch = np.nan

    try:
        db = davies_bouldin_score(embedding, labels)
    except ValueError:
        db = np.nan

    return {
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
    }


def neighborhood_purity(
    coords: np.ndarray, labels: np.ndarray, k: int = 6
) -> float:
    """
    Compute spatial neighborhood purity: mean fraction of k-nearest neighbors
    that share the same cluster label as each spot.

    Args:
        coords: 2D array of shape (n_samples, 2) - spatial coordinates.
        labels: 1D array of cluster labels for each sample.
        k: Number of neighbors to consider.

    Returns:
        Mean neighborhood purity across all spots (float between 0 and 1).
    """
    # Convert labels to numpy array
    if hasattr(labels, "cat"):
        labels = labels.cat.codes.values
    labels = np.asarray(labels)

    n_samples = coords.shape[0]
    if n_samples <= k:
        k = max(1, n_samples - 1)

    # Build kNN graph over spatial coordinates (k+1 to include self)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Exclude self (first neighbor is the point itself)
    neighbor_indices = indices[:, 1:]

    # Compute purity for each spot
    purities = np.zeros(n_samples)
    for i in range(n_samples):
        neighbor_labels = labels[neighbor_indices[i]]
        same_label_count = np.sum(neighbor_labels == labels[i])
        purities[i] = same_label_count / k

    return float(np.mean(purities))


def compute_sctype_stability(
    adata: anndata.AnnData,
    sctype_keys: list[str] = ["sctype_pca", "sctype_svae", "sctype_cvae"],
) -> tuple[pd.Series, float]:
    """
    Compute sctype annotation stability across different embeddings.

    For each spot, computes the majority agreement fraction across all available
    sctype columns. E.g., if 3 columns exist and all 3 agree -> 1.0,
    if 2 of 3 agree -> 2/3, all different -> 1/3.

    Args:
        adata: AnnData object with sctype annotation columns in obs.
        sctype_keys: List of column names to compare. Missing columns are skipped.

    Returns:
        Tuple of (per_spot_stability Series, mean_stability scalar).
        Returns (empty Series, NaN) if fewer than 2 sctype columns are present.
    """
    # Filter to columns that exist
    available_keys = [k for k in sctype_keys if k in adata.obs.columns]

    if len(available_keys) < 2:
        return pd.Series(dtype=float), np.nan

    n_methods = len(available_keys)
    n_spots = adata.n_obs

    # Build matrix of sctype labels
    sctype_matrix = adata.obs[available_keys].values

    per_spot_stability = np.zeros(n_spots)
    for i in range(n_spots):
        labels_at_spot = sctype_matrix[i, :]
        # Count occurrences of each label
        label_counts = Counter(labels_at_spot)
        # Majority count
        max_count = max(label_counts.values())
        per_spot_stability[i] = max_count / n_methods

    stability_series = pd.Series(per_spot_stability, index=adata.obs_names, name="sctype_stability")
    mean_stability = float(np.mean(per_spot_stability))

    return stability_series, mean_stability


def evaluate_embeddings(
    adata: anndata.AnnData,
    spatial_k: int = 6,
) -> pd.DataFrame:
    """
    Evaluate all available embeddings (PCA, sVAE, cVAE) with internal cluster
    metrics and spatial neighborhood purity.

    This function gracefully skips methods whose embeddings or cluster labels
    are missing from adata.

    Args:
        adata: AnnData object with embeddings in obsm and cluster labels in obs.
               Expected keys:
               - Embeddings: X_pca, X_spatial_vae, X_cvae
               - Labels: leiden_pca, leiden_svae, leiden_cvae
        spatial_k: Number of neighbors for neighborhood purity calculation.

    Returns:
        DataFrame with rows for each available method and columns:
        method, silhouette, calinski_harabasz, davies_bouldin, neighborhood_purity.
    """
    # Define method configurations
    method_configs = [
        {"name": "PCA", "embedding_key": "X_pca", "label_key": "leiden_pca"},
        {"name": "sVAE", "embedding_key": "X_spatial_vae", "label_key": "leiden_svae"},
        {"name": "cVAE", "embedding_key": "X_cvae", "label_key": "leiden_cvae"},
    ]

    coords = adata.obsm["spatial"]
    results = []

    for cfg in method_configs:
        embedding_key = cfg["embedding_key"]
        label_key = cfg["label_key"]
        method_name = cfg["name"]

        # Check if both embedding and labels exist
        if embedding_key not in adata.obsm:
            continue
        if label_key not in adata.obs:
            continue

        embedding = adata.obsm[embedding_key]
        labels = adata.obs[label_key]

        # Compute internal metrics
        metrics = compute_internal_cluster_metrics(embedding, labels)

        # Compute spatial neighborhood purity
        purity = neighborhood_purity(coords, labels, k=spatial_k)

        results.append({
            "method": method_name,
            "silhouette": metrics["silhouette"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "davies_bouldin": metrics["davies_bouldin"],
            "neighborhood_purity": purity,
        })

    return pd.DataFrame(results)


def evaluate_sctype_across_resolutions(
    adata: anndata.AnnData,
    embedding_key: str,
    method_prefix: str,
    resolutions: list[float],
    run_sctype_fn,
    tissue_type: str = "Breast",
    db: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate sctype robustness across multiple Leiden resolutions for a single embedding.

    For each resolution:
    - Re-runs Leiden clustering
    - Runs sctype annotation
    - Stores results in adata.obs with resolution-specific keys

    Then computes:
    - Per-spot sctype stability across resolutions
    - AMI between sctype labelings at different resolutions (pairwise)

    Args:
        adata: AnnData object with the embedding in obsm.
        embedding_key: Key in adata.obsm for the embedding (e.g., "X_spatial_vae").
        method_prefix: Prefix for naming (e.g., "svae" -> "leiden_svae_res0_4").
        resolutions: List of Leiden resolutions to test.
        run_sctype_fn: The run_sctype function from sctypepy.
        tissue_type: Tissue type for sctype.
        db: Custom database DataFrame for sctype.

    Returns:
        Tuple of:
        - DataFrame with resolution-level metrics
        - Dict with summary stats including mean stability and pairwise AMI matrix
    """
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")

    # Build neighbors graph if needed
    sc.pp.neighbors(adata, use_rep=embedding_key)

    sctype_keys = []
    resolution_results = []

    for res in resolutions:
        # Create resolution-specific key names (replace . with _)
        res_str = str(res).replace(".", "_")
        leiden_key = f"leiden_{method_prefix}_res{res_str}"
        sctype_key = f"sctype_{method_prefix}_res{res_str}"

        # Run Leiden clustering at this resolution
        sc.tl.leiden(adata, resolution=res, key_added=leiden_key)

        # Run sctype annotation
        adata = run_sctype_fn(adata, tissue_type=tissue_type, groupby=leiden_key, db=db)
        adata.obs[sctype_key] = adata.obs["sctype_classification"]
        sctype_keys.append(sctype_key)

        # Compute cluster metrics for this resolution
        embedding = adata.obsm[embedding_key]
        labels = adata.obs[leiden_key]
        metrics = compute_internal_cluster_metrics(embedding, labels)

        resolution_results.append({
            "resolution": res,
            "leiden_key": leiden_key,
            "sctype_key": sctype_key,
            "n_clusters": len(labels.unique()),
            **metrics,
        })

    resolution_df = pd.DataFrame(resolution_results)

    # Compute stability across resolutions
    stability_series, mean_stability = compute_sctype_stability(adata, sctype_keys=sctype_keys)

    # Compute pairwise AMI between sctype labelings
    n_res = len(resolutions)
    ami_matrix = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            labels_i = adata.obs[sctype_keys[i]].astype(str)
            labels_j = adata.obs[sctype_keys[j]].astype(str)
            ami_matrix[i, j] = adjusted_mutual_info_score(labels_i, labels_j)

    summary = {
        "mean_sctype_stability_across_resolutions": mean_stability,
        "ami_matrix": ami_matrix.tolist(),
        "resolutions": resolutions,
    }

    return resolution_df, summary


def save_evaluation_results(
    metrics_df: pd.DataFrame,
    sctype_stability: float,
    output_prefix: str = "cluster_metrics",
    output_dir: str = "results",
) -> None:
    """
    Save evaluation results to disk.

    Args:
        metrics_df: DataFrame from evaluate_embeddings.
        sctype_stability: Mean sctype stability score.
        output_prefix: Prefix for output files.
        output_dir: Directory to save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics DataFrame
    csv_path = output_path / f"{output_prefix}.csv"
    metrics_df.to_csv(csv_path, index=False)

    # Save summary including sctype stability
    summary = {
        "sctype_mean_stability": sctype_stability,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    json_path = output_path / f"{output_prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   Saved metrics to: {csv_path}")
    print(f"   Saved summary to: {json_path}")

