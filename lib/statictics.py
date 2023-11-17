import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def calculate_center(df: pd.DataFrame, clusters_col_name: str) -> pd.DataFrame:
    df_clusters = df[clusters_col_name].value_counts().reset_index()
    df_clusters.columns = ['cluster_id', 'cluster_size']

    centers, max_distances, mean_distances = [], [], []
    for cluster_id in df_clusters.cluster_id:
        cluster_phrases = df.loc[df[clusters_col_name] == cluster_id, 'one_name'].values
        cluster_embeddings = df.loc[df[clusters_col_name] == cluster_id, 'ft_vectors']

        dist_matrix = compute_dist_matrix(cluster_embeddings, metric='inner_product')
        mean_point_distance = dist_matrix.mean(axis=1)
        center_idx = mean_point_distance.argmin()

        centers.append(cluster_phrases[center_idx])

    df_clusters['cluster_center'] = centers
    return df_clusters


def compute_dist_matrix(X: np.ndarray, metric: str = 'inner_product') -> np.ndarray:
    if X.ndim == 1:
        X = X[None, :]

    if metric == 'inner_product':
        dist_matrix = 1 - np.inner(X, X)
    else:
        dist_matrix = squareform(pdist(X, metric=metric))

    return dist_matrix
