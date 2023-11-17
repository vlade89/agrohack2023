""" Text Preprocessing """
import logging
import re
from functools import lru_cache
from multiprocessing import Pool
from typing import Optional, List

import numpy as np
import pandas as pd
import pymorphy2
from stop_words import get_stop_words

wcoll_morph: Optional[pymorphy2.MorphAnalyzer] = None
g_chunks: Optional[List[pd.DataFrame]] = None


def ensure_morph():
    global wcoll_morph
    if wcoll_morph is None:
        wcoll_morph = pymorphy2.MorphAnalyzer()


def release_morph():
    global wcoll_morph
    wcoll_morph = None


def make_chuncks(df, proc_count):
    result = []
    data_size = df.shape[0]
    chunksize = int(np.ceil(data_size / proc_count))
    left = right = 0
    for i in range(proc_count - 1):
        right += chunksize
        result.append(df[left:right])
        left += chunksize
    result.append(df[left:])
    return result


class Preprocessing:
    """ Clean, tokenize and normalize texts """

    def __init__(self):
        self.stopwords = set(get_stop_words("ru"))
        self.pattern = re.compile("[А-Яа-яA-z0-9]+")

    def process_texts(self, df, text_col, proc_count=1):
        ensure_morph()
        df["tokens"] = df[text_col].str.lower().str.findall(self.pattern)
        try:
            if proc_count == 1:
                df["tokens"] = df["tokens"].apply(
                    lambda txt: self.tokenize(txt)
                )
            else:
                global g_chunks
                logging.info("Reading chunks ...")
                g_chunks = list(make_chuncks(df["tokens"], proc_count=proc_count))
                logging.info(
                    "Chunk count %s %s",
                    len(g_chunks),
                    sum(ch.shape[0] for ch in g_chunks),
                )
                logging.info("Processing chunks ...")
                with Pool(proc_count) as p:
                    result = p.map(self.process_series, range(len(g_chunks)))
                df["tokens"] = pd.concat(result)
                g_chunks = None
        finally:
            release_morph()
        return df["tokens"]

    @lru_cache(maxsize=50000)
    def normalize_word(self, token):
        """
        Pymorphy2 normalizer.

        Args:
            token: str
                token to normalize
        Returns:
            str
        """
        global wcoll_morph

        return wcoll_morph.parse(token)[0].normal_form

    def tokenize(self, arr):
        """
        Tokenizes, normalizes input text, removes stop-words.

        Args:
            arr: List[str]
                list of tokens
        Returns:
            list of integers
        """
        return [
            self.normalize_word(t.strip())
            for t in arr
            if t not in self.stopwords and len(t) > 2
        ]

    def process_series(self, chunk_num: int) -> pd.Series:
        global g_chunks
        return g_chunks[chunk_num].apply(lambda txt: self.tokenize(txt))