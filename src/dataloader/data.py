import numpy as np
import torch
import scanpy as sc
import squidpy as sq
from sklearn.neighbors import NearestNeighbors


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
    
    # Get k-nearest neighbors directly to preserve distance-based ordering
    distances, indices = knn_model.kneighbors(coords)
    
    # Build edge index by iterating through each node and its neighbors
    src_list = []
    dst_list = []
    
    for i in range(len(coords)):
        for j in indices[i]:
            # Skip self-loops
            if i != j:
                src_list.append(i)
                dst_list.append(j)
    
    # Convert to torch tensors
    src_tensor = torch.tensor(src_list, dtype=torch.long)
    dst_tensor = torch.tensor(dst_list, dtype=torch.long)
    edge_index = torch.stack([src_tensor, dst_tensor], dim=0)
    
    return edge_index
