""" 
    File Name:          DLTM/protien_seq_counting.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/28/18
    Python Version:     3.6.6
    File Description:   

"""
import itertools
import multiprocessing
import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def get_feature_list(
        token_len: int,
        protein_seqs: iter):

    token_set = set([])

    for p in protein_seqs:

        for i in range(len(p)):

            if i + token_len > len(p):
                pass
            else:
                token_set.add(p[i: i + token_len])

    return sorted(list(token_set))


def count_protein_seq(
        data_path: str,
        token_len: int,
        feature_list: iter,
        protein_seqs: iter,
        count_strat: str):

    assert count_strat in ['occurrence', 'frequency']

    # If the data already exist, load and return
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    feature_list = sorted(feature_list)

    # Iterate through all the protein sequences
    # Parallelized version
    def featurize_protein_seq(p):

        featurized_p = np.zeros(len(feature_list)).astype(np.int16)
        for i in range(0, len(p) - token_len + 1):
            if p[i: i + token_len] in feature_list:
                featurized_p[feature_list.index(p[i: i + token_len])] += 1

        if count_strat == 'frequency':
            featurized_p = np.array(
                [(occ / (len(p) - len(feature_list[idx]) + 1))
                 for idx, occ in enumerate(featurized_p)]).astype(np.float16)

        return featurized_p

    num_cores = multiprocessing.cpu_count()
    featurized_protein_seqs = Parallel(n_jobs=num_cores)(
        delayed(featurize_protein_seq)(p) for p in protein_seqs)

    with open(data_path, 'wb') as f:
        pickle.dump(featurized_protein_seqs, f)

    return featurized_protein_seqs


if __name__ == '__main__':

    trn_dataframe = pd.read_csv('../../data/coreseed.train.tsv',
                                sep='\t', usecols=['protein', 'function'])
    trn_proteins = trn_dataframe['protein']

    val_dataframe = pd.read_csv('../../data/coreseed.test.tsv',
                                sep='\t', usecols=['protein', 'function'])
    val_proteins = val_dataframe['protein']

    for l in [ 3]:

        f_list = get_feature_list(l, trn_proteins)

        for count_strat in ['occurrence', 'frequency']:

            count_protein_seq(
                data_path='../../data/CoreSEED_trn_featurized_on_%s(%i).pkl'
                          % (count_strat, l),
                token_len=l,
                feature_list=f_list,
                protein_seqs=trn_proteins,
                count_strat=count_strat)

            count_protein_seq(
                data_path='../../data/CoreSEED_val_featurized_on_%s(%i).pkl'
                          % (count_strat, l),
                token_len=l,
                feature_list=f_list,
                protein_seqs=val_proteins,
                count_strat=count_strat)
