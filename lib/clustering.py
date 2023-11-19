import logging
import os
import random
import re

import compress_fasttext
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score

from lib.nlp_utils import Preprocessing
from lib.read_data import parse_resume, read_resume
from lib.statictics import calculate_center

logging.basicConfig(level=logging.INFO)


def build_clusters(
        input_data_path: str,
        output_data_path: str,
        model_path: str,
        max_n_clusters: int = 100,
        min_n_clusters: int = 2
) -> pd.DataFrame:
    """
    Loads, processes and vectorizes the data. Clusterizes embeddings and culculates clusters' centers
    for each number of clusters in range from 'min_n_clusters' to 'max_n_clusters'.

    Args:
        input_data_path: a path to the folder with input data
        output_data_path: a path to the folder with output data
        model_path: a path to the vectorizer
        max_n_clusters: the maximum number of cluster
        min_n_clusters: the minimum number of cluster

    Returns:
        A table including the clusters' labels and clusters' centers
        for each number of clusters in range from 'min_n_clusters' to 'max_n_clusters'.
    """

    # loading FastText vectors
    vectorizer = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)

    # setting random state for reproducibility
    seed = 2023
    random.seed(seed)

    # reading html files
    resumes = []
    for file in os.listdir(input_data_path):
        my_soup = read_resume(file)
        resumes.append(parse_resume(my_soup))
    logging.info('Total number of resumes: ', len(resumes))

    # selecting only one if several specialties are specified
    data = (
        pd.DataFrame
        .from_records(resumes)
        .assign(
            one_name=lambda df: df['name'].apply(lambda txt: re.split('[,/.]', txt)[0].strip('./!? '))
        )
        .reset_index()
        .rename(columns={'index': 'id'})
    )
    logging.info('Data shape:', data.shape)

    # cleaning, tokenizing and vectorizing
    logging.info('Preprecessing texts...')
    proc = Preprocessing()
    data['tokens_for_clustering'] = proc.process_texts(data, 'one_name')
    clustered_data = (
        data
        .loc[data['tokens_for_clustering'].apply(lambda x: len(x) != 0)]
        .assign(
            ft_vectors=lambda df: df['tokens_for_clustering'].apply(
                lambda txt: np.array([vectorizer[token] for token in txt]).mean(axis=0)
            )
        )
    )[['id', 'one_name', 'ft_vectors']]

    # transferring to numpy
    ft_vectors = np.concatenate(
        clustered_data['ft_vectors'].values
    ).reshape(clustered_data.shape[0], -1)

    # performing clustering in a loop
    logging.info('Making clusterisation...')
    output_dfs = []
    for num in range(min_n_clusters, max_n_clusters):
        cl = cluster.AgglomerativeClustering(n_clusters=num)
        tmp_clustered = clustered_data.copy()

        tmp_clustered['num_clusters'], tmp_clustered['cluster_id'] = num, cl.fit_predict(ft_vectors)
        score = silhouette_score(ft_vectors, cl.labels_)

        stats = calculate_center(tmp_clustered, 'cluster_id', num)
        tmp_clustered = tmp_clustered.merge(
            stats,
            how='left',
            on=['cluster_id'],
            suffixes=[None, '_y']
        )
        del tmp_clustered['num_clusters_y']

        tmp_clustered['score'] = score

        output_dfs.append(tmp_clustered)

    output_data = pd.concat(output_dfs)

    data.to_csv(output_data_path + 'resumes.csv', index=False)
    # clustered_data.to_csv('clustered_data.csv')
    output_data.to_csv('clustered_data.csv')
    logging.info('Resulting data shape: %s', clustered_data.shape)
    logging.info('Columns: %s', clustered_data.columns)
    logging.info('First row: %s', clustered_data.iloc[0, :])

    return output_data
