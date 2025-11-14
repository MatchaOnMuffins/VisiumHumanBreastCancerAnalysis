import numpy as np
import torch
import scanpy as sc
import squidpy as sq
from sklearn.neighbors import NearestNeighbors

NORMALIZATION_TARGET_SUM = 1e4


def load_visium_data(data_dir=".", counts_file="Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"):
    """Load Visium spatial transcriptomics data."""
    return sq.read.visium(data_dir, counts_file=counts_file)


def filter_genes_by_spot_count(expression_matrix, min_spots):
    genes_expressed_mask = (expression_matrix > 0).sum(axis=0) > min_spots
    return expression_matrix[:, genes_expressed_mask]


def normalize_and_log_transform(expression_matrix, target_sum):
    adata = sc.AnnData(expression_matrix)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_highly_variable_genes(adata, n_top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    return adata[:, adata.var['highly_variable']]


def compute_pca_features(adata, n_components):
    sc.tl.pca(adata, n_comps=n_components)
    return adata.obsm["X_pca"].astype(np.float32)


def preprocess_expression(X, min_spots=200, n_top_genes=3000, n_pca_components=50):
    """Preprocess gene expression: filter, normalize, select variable genes, run PCA."""
    filtered_expression = filter_genes_by_spot_count(X, min_spots)
    adata = normalize_and_log_transform(filtered_expression, NORMALIZATION_TARGET_SUM)
    adata = select_highly_variable_genes(adata, n_top_genes)
    return compute_pca_features(adata, n_pca_components)


def create_knn_adjacency_matrix(coordinates, k):
    knn_model = NearestNeighbors(n_neighbors=k).fit(coordinates)
    return knn_model.kneighbors_graph(coordinates, mode="connectivity")


def convert_adjacency_to_edge_index(adjacency_matrix):
    sparse_coo = adjacency_matrix.tocoo()
    row_indices = torch.from_numpy(sparse_coo.row).long()
    col_indices = torch.from_numpy(sparse_coo.col).long()
    return torch.stack([row_indices, col_indices], dim=0)


def remove_self_loops(edge_index):
    is_not_self_loop = edge_index[0] != edge_index[1]
    return edge_index[:, is_not_self_loop]


def build_spatial_graph(coords, k=6):
    """Build k-NN graph from spatial coordinates, return edge_index tensor."""
    adjacency_matrix = create_knn_adjacency_matrix(coords, k)
    edge_index = convert_adjacency_to_edge_index(adjacency_matrix)
    return remove_self_loops(edge_index)
