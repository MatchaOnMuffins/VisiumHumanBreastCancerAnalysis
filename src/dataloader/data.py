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

def annotate_cell_types(adata, tissue="Kidney"):
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    adata = run_sctype(adata, tissue_type=tissue, groupby="leiden")
    return None

def compute_cell_type_fractions(adata, k=6):
    labels = adata.obs["sctype_classification"].values
    unique_labels = np.unique(labels)

    nbrs = NearestNeighbors(n_neighbors=k).fit(adata.obsm["spatial"])
    _, indices = nbrs.kneighbors(adata.obsm["spatial"])

    neighbor_fractions = []
    for i in range(adata.n_obs):
        neighbor_labels = labels[indices[i]]
        fractions = [
            np.sum(neighbor_labels == ulab) / k for ulab in unique_labels
        ]
        neighbor_fractions.append(fractions)

    return np.array(neighbor_fractions), unique_labels

def create_spatial_features(adata, k=6):
    x_coords = adata.obsm["spatial"][:, 0]
    y_coords = adata.obsm["spatial"][:, 1]

    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    r_norm = np.sqrt(x_norm**2 + y_norm**2)

    neighbor_comp, unique_labels = compute_cell_type_fractions(adata, k=k)

    spatial_features = np.column_stack([x_norm, y_norm, r_norm, neighbor_comp])
    feature_names = (
        ["x_norm", "y_norm", "r_norm"] +
        [f"nbr_frac_{lab}" for lab in unique_labels]
    )

    return spatial_features, feature_names
