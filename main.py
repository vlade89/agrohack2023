""" Точка входа. """

from lib.clustering import build_clusters


build_clusters('data/', '.', 'nlp_model/small_model_11_18', max_n_clusters=100)
