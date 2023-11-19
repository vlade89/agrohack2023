import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def calculate_center(df: pd.DataFrame, clusters_col_name: str, num: int) -> pd.DataFrame:
    df_clusters = (
        df
        .groupby(clusters_col_name, as_index=False)
        .agg(
            cluster_size=('id', 'count'),
            cluster_mode=('one_name', lambda s: s.mode()[0]),
        )
    )

    centers, vectors = [], []
    for cluster_id in df_clusters[clusters_col_name]:
        cluster_phrases = df.loc[df[clusters_col_name] == cluster_id, 'one_name'].values
        cluster_embeddings = df.loc[df[clusters_col_name] == cluster_id, 'ft_vectors'].values

        dist_matrix = compute_dist_matrix(cluster_embeddings, metric='inner_product')
        mean_point_distance = dist_matrix.mean(axis=1)
        center_idx = mean_point_distance.argmin()

        centers.append(cluster_phrases[center_idx])
        vectors.append(cluster_embeddings[center_idx])

    df_clusters['cluster_center'] = centers
    df_clusters['cluster_center_vector'] = vectors
    df_clusters.columns = [
        clusters_col_name,
        'cluster_size',
        'cluster_mode',
        'cluster_center',
        'cluster_center_vector'
    ]

    df_clusters['num_clusters'] = num

    return df_clusters


def compute_dist_matrix(mtrx: np.ndarray, metric: str = 'inner_product') -> np.ndarray:
    if mtrx.ndim == 1:
        mtrx = mtrx[None, :]

    if metric == 'inner_product':
        dist_matrix = 1 - np.inner(mtrx, mtrx)
    else:
        dist_matrix = squareform(pdist(mtrx, metric=metric))

    return dist_matrix
