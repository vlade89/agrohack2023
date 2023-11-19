import os
import sys
import random
import re
import logging

import compress_fasttext
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score

from nlp_utils import Preprocessing
from read_data import parse_resume, read_resume
from statictics import calculate_center

logging.basicConfig(level=logging.INFO)


def build_clusters(input_data_path, output_data_path, model_path, max_n_clusters=100):
    vectorizer = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)
    seed = 2023
    random.seed(seed)

    resumes = []
    for file in os.listdir(input_data_path):
        my_soup = read_resume(file)
        resumes.append(parse_resume(my_soup))
    logging.info('Total number of resumes: ', len(resumes))

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

    ft_vectors = np.concatenate(
        clustered_data['ft_vectors'].values
    ).reshape(clustered_data.shape[0], -1)

    logging.info('Making clusterisation...')
    output_dfs = []
    for num in range(2, max_n_clusters):
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


build_clusters('data/', '.', 'nlp_model/small_model_11_18', max_n_clusters=100)
