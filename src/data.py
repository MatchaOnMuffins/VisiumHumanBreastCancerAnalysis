import numpy as np
import torch
import scanpy as sc
import squidpy as sq
from sklearn.neighbors import NearestNeighbors


def load_visium_data(data_dir=".", counts_file="Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"):
    return sq.read.visium(data_dir, counts_file=counts_file)


def preprocess_expression(X, min_spots=200, n_top_genes=3000, n_pca_components=50):
    genes_expressed_mask = (X > 0).sum(axis=0) > min_spots
    filtered_expression = X[:, genes_expressed_mask]

    adata = sc.AnnData(filtered_expression)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]

    sc.tl.pca(adata, n_comps=n_pca_components)
    return adata.obsm["X_pca"].astype(np.float32)


def build_spatial_graph(coords, k=6):
    knn_model = NearestNeighbors(n_neighbors=k).fit(coords)
    adjacency_matrix = knn_model.kneighbors_graph(coords, mode="connectivity")

    sparse_coo = adjacency_matrix.tocoo() # type: ignore
    row_indices = torch.from_numpy(sparse_coo.row).long()
    col_indices = torch.from_numpy(sparse_coo.col).long()
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    is_not_self_loop = edge_index[0] != edge_index[1]
    return edge_index[:, is_not_self_loop]
