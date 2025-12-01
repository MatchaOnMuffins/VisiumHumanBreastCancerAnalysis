import numpy as np
import torch
import scanpy as sc
import squidpy as sq
from sklearn.neighbors import NearestNeighbors
from sctypepy import run_sctype



def load_visium_data(
    data_dir=".", counts_file="Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
):
    return sq.read.visium(data_dir, counts_file=counts_file)


def preprocess_expression(X, min_spots=200, n_top_genes=3000, n_pca_components=50):
    genes_expressed_mask = (X > 0).sum(axis=0) > min_spots
    filtered_expression = X[:, genes_expressed_mask]

    adata = sc.AnnData(filtered_expression)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]]

    sc.tl.pca(adata, n_comps=n_pca_components)
    return adata.obsm["X_pca"].astype(np.float32)


def build_spatial_graph(coords, k=6):
    knn_model = NearestNeighbors(n_neighbors=k).fit(coords)
    adjacency_matrix = knn_model.kneighbors_graph(coords, mode="connectivity")

    sparse_coo = adjacency_matrix.tocoo()  # type: ignore
    row_indices = torch.from_numpy(sparse_coo.row).long()
    col_indices = torch.from_numpy(sparse_coo.col).long()
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    is_not_self_loop = edge_index[0] != edge_index[1]
    return edge_index[:, is_not_self_loop]

def annotate_unsupervised_domains(adata, n_pcs=50, resolution=0.5):
    """
    Runs PCA and Leiden clustering to produce unsupervised expression domains.
    Stores them in adata.obs['expr_cluster'] as integer cluster IDs.
    """

    # 1. PCA (assuming adata is preprocessed; otherwise add HVG/log1p)
    sc.pp.pca(adata, n_comps=n_pcs)

    # 2. Build KNN graph in PCA space
    sc.pp.neighbors(adata, use_rep="X_pca")

    # 3. Unsupervised clustering in PCA space
    sc.tl.leiden(adata, resolution=resolution, key_added="expr_cluster")

    # Convert to integer codes for easy indexing
    adata.obs["expr_cluster"] = adata.obs["expr_cluster"].astype("category")
    adata.obs["expr_cluster_id"] = adata.obs["expr_cluster"].cat.codes

    return adata

def compute_clusterID_fractions(adata, k=6):
    """
    Computes fractional composition of unsupervised cluster IDs
    in the spatial neighborhood of each spot.
    """

    # Unsupervised cluster IDs (integer codes)
    labels = adata.obs["expr_cluster_id"].values
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # kNN graph in spatial coordinates
    coords = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Compute fraction vectors
    neighbor_fractions = np.zeros((adata.n_obs, n_clusters))

    for i in range(adata.n_obs):
        neighbor_labels = labels[indices[i]]
        for j, lab in enumerate(unique_labels):
            neighbor_fractions[i, j] = np.sum(neighbor_labels == lab) / k

    return neighbor_fractions, unique_labels

def create_spatial_features(adata, k=6):
    """
    Creates the final spatial conditioning feature matrix for the cVAE
    using unsupervised clusterID neighbor fractions.
    """

    # 1. Get fractional cluster composition of spatial neighbors
    neighbor_comp, unique_labels = compute_clusterID_fractions(adata, k=k)

    # 2. Spatial features = just the fractional composition vector
    spatial_features = neighbor_comp

    # 3. Nice naming
    feature_names = [f"unsupClusterFrac_{lab}" for lab in unique_labels]

    return spatial_features, feature_names
