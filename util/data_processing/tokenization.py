""" 
    File Name:          DLTM/tokenization.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/7/18
    Python Version:     3.6.6
    File Description:   

"""

import os
import json
import pickle
import logging
import time

import numpy as np
import pandas as pd
import multiprocessing
from rdkit import Chem
from joblib import Parallel, delayed, parallel_backend
from deepchem.molnet import load_hiv, load_pcba

from util.misc.path_creating import create_path

logger = logging.getLogger(__name__)


# Default special tokens for SMILES strings, SMARTS string and protein seq
SP_TOKENS = {
    'SOS': '<SOS>',
    'EOS': '<EOS>',
    'UNK': '<UNK>',
    'PAD': '<PAD>',
}


def get_smiles_token_dict(
        dict_path: str,
        tokenize_on: str,
        smiles_strings: iter,
        reload_from_disk: bool = True):
    """token_dict = get_token_dict('./dict_path', smiles_strings, 'char')

    This function reads of a list of SMILES strings (or any string based)
    and constructs an encoding/tokenization dictionary if it does not
    currently exist in the given path.

    WARNING: when using 'atom' tokenization, the given sentences must be
    valid SMILES strings acceptable for RDKit.

    WARNING: special tokens should not share the same prefix as any other
        tokens in order to maintain the uniqueness of tokenization.

    Args:
        dict_path (str): path to the tokenization dictionary
        smiles_strings (iter): an iterable structure of SMILES to be tokenized
        tokenize_on (str): tokenization method. Choose between 'atom'
            (tokenize on atoms/bonds)and 'char' (tokenize on characters)
        reload_from_disk (bool): indicator for saving and reloading the
            tokenization dictionary from disk

    Returns:
        dict: sorted dictionary for tokenization
    """

    # Load the tokenization dictionary if it exists already
    if os.path.exists(dict_path) and reload_from_disk:
        with open(dict_path, 'r') as f:
            return json.load(f)

    # Make sure of the tokenize strategy, data path, and special tokens
    assert tokenize_on in ['char', 'atom']
    create_path(os.path.dirname(dict_path))

    special_tokens = SP_TOKENS

    # Create the tokenization dictionary based on the chars
    if tokenize_on == 'char':
        tokens = set.union(*[set(s) for s in smiles_strings])

    # Otherwise, create the tokenization dictionary based on atoms/bonds
    else:

        # Transfer everything into SMARTS strings for better tokenization
        try:
            # smarts_strings = []
            # for s in smiles_strings:
            #     mol = Chem.MolFromSmiles(s)
            #     if mol:
            #         smarts_strings.append(Chem.MolToSmarts(mol))

            molecules = [Chem.MolFromSmiles(s) for s in smiles_strings]
            smarts_strings = [Chem.MolToSmarts(m) for m in molecules if m]

            assert len(molecules) == len(smarts_strings)

        except:
            logger.error('Failed to parse given SMILES strings into SMARTS, '
                         'Make sure that the SMILES strings are valid and '
                         'acceptable for RDKit, or use \'char\' tokenization.')
            raise

        # Iterate through the SMARTS strings
        # Things inside bracket should be considered a single token
        tokens = set()

        for s in smarts_strings:

            skip_char = 0
            for i in range(len(s)):

                if skip_char > 0:
                    skip_char = skip_char - 1
                    continue

                if s[i] == '[':

                    j = s.find(']', i + 1)

                    # Make sure that there is no nested bracket
                    assert j < s.find('[', i + 1) or s.find('[', i + 1) == -1

                    tokens.add(s[i: j + 1])
                    skip_char = j - i

                else:
                    tokens.add(s[i])

    # Add special tokens into the token list
    # After making sure that they are have completely exclusive elements
    assert set(special_tokens.values()).isdisjoint(tokens)
    tokens = tokens.union(set(special_tokens.values()))

    # Check for the validity of token dictionary
    # That is, no token should be the start of the other tokens
    for t1 in tokens:
        assert all(((not t1.startswith(t2)) or (t1 == t2)) for t2 in tokens)

    # Generates ordered token dictionary for better readability, then save
    token_dict = dict((c, i) for i, c in enumerate(sorted(tokens)))
    if reload_from_disk:
        with open(dict_path, 'w') as f:
            json.dump(token_dict, f, indent=4, separators=(',', ': '))

    return token_dict


def tokenize_smiles(
        data_path: str,
        token_dict: dict,
        smiles_strings: iter,
        max_seq_length: int = 128,
        reload_from_disk: bool = True):
    """smiles, tokenized_smiles = \
        tokenize_smiles('./data_path', token_dict, smiles_strings)

    This function tokenizes given SMILES strings with given token
    dictionary. Before returning the tokenized strings, it also pads every
    strings into the same length and gets rid of the strings that exceed
    max_seq_length after tokenization.

    Args:
        data_path (str): path to the tokenized data
        token_dict (dict): sorted dictionary for tokenization
        smiles_strings (iter): an iterable structure of SMILES to be tokenized
        max_seq_length (int): maximum sequence length for tokenized strings
        reload_from_disk (bool): indicator for saving and reloading the
            tokenized data from disk

    Returns:
        (
            list: SMILES strings within max_seq_length after tokenization
            list: tokenized and padded SMILES strings from previous list
        )
    """

    special_tokens = SP_TOKENS

    # Load the tokenized SMILES strings if the data file exists
    if os.path.exists(data_path) and reload_from_disk:
        with open(data_path, 'rb') as f:
            tokenized_smiles_strings = pickle.load(f)

    else:
        # Make sure of the data directory
        create_path(os.path.dirname(data_path))

        # For each SMILES string:
        # 1. Add "start of the sentence" token at the beginning
        # 2. Tokenize the next words (of various length)
        # 3. End the list with the "end of the sentence" token

        # Use joblib for parallelization
        sos_token = token_dict[special_tokens['SOS']]
        eos_token = token_dict[special_tokens['EOS']]

        def tokenize_one_smiles_string(s):

            # Transfer to SMARTS string before tokenization
            molecule = Chem.MolFromSmiles(s)
            smart_string = Chem.MolToSmarts(molecule)

            curr_index = 0
            t = [sos_token, ]

            while curr_index != len(smart_string):
                for token in token_dict:
                    if smart_string[curr_index:].startswith(token):
                        t.append(token_dict[token])
                        curr_index += len(token)

            t.append(eos_token)
            return t

        num_cores = multiprocessing.cpu_count()
        tokenized_smiles_strings = Parallel(n_jobs=num_cores)(
            delayed(tokenize_one_smiles_string)(s) for s in smiles_strings)

        # Serialize way of tokenization
        # tokenized_smiles_strings = []
        # for s in smiles_strings:
        #
        #     # Transfer to SMARTS string before tokenization
        #     molecule = Chem.MolFromSmiles(s)
        #     smart_string = Chem.MolToSmarts(molecule)
        #
        #     curr_index = 0
        #     t = [token_dict[special_tokens['SOS']], ]
        #
        #     while curr_index != len(s):
        #         for token in token_dict:
        #             if s[curr_index:].startswith(token):
        #                 t.append(token_dict[token])
        #                 curr_index += len(token)
        #
        #     t.append(token_dict[special_tokens['EOS']])
        #     tokenized_smiles_strings.append(t)

        # Write the tokenized SMILES strings into file (without padding)
        if reload_from_disk:
            with open(data_path, 'wb') as f:
                pickle.dump(tokenized_smiles_strings, f)

    # Take care of padding and only return the strings within given length
    if max_seq_length is None:
        max_seq_length = 0
        for t in tokenized_smiles_strings:
            if len(t) > max_seq_length:
                max_seq_length = len(t)

    ret_smiles_strings = \
        [s for s, t in zip(smiles_strings, tokenized_smiles_strings)
         if len(t) <= max_seq_length]
    ret_tokenized_smiles_strings = \
        [t + [token_dict[special_tokens['PAD']], ] * (max_seq_length - len(t))
         for t in tokenized_smiles_strings if len(t) <= max_seq_length]

    logger.warning('Keeping %i out of %i (%.2f%%) SMILES strings due to '
                   'maximum length limitation.'
                   % (len(ret_smiles_strings), len(smiles_strings),
                      100. * len(ret_smiles_strings)/len(smiles_strings)))

    return ret_smiles_strings, ret_tokenized_smiles_strings


def get_protein_token_dict(
        dict_path: str,
        token_length: int,
        protein_seqs: iter,
        reload_from_disk: bool = True):
    """token_dict = get_protein_token_dict( './dict_path', protein_seqs)

    :param dict_path:
    :param token_length:
    :param protein_seqs:
    :param reload_from_disk:
    :return:
    """

    # Load the tokenization dictionary if it exists already
    if os.path.exists(dict_path) and reload_from_disk:
        with open(dict_path, 'r') as f:
            return json.load(f)

    # Make sure of the tokenization length, data path, and special tokens
    assert token_length > 0
    create_path(os.path.dirname(dict_path))
    special_tokens = SP_TOKENS

    total_seq_length = 0
    occurr_dict = {}
    for ps in protein_seqs:

        total_seq_length += len(ps)
        for i in range(len(ps)):
            for j in range(1, token_length + 1):

                if i + j + 1 <= len(ps):

                    sub_seq = ps[i: i + j]

                    if sub_seq in occurr_dict:
                        occurr_dict[sub_seq] = occurr_dict[sub_seq] + 1
                    else:
                        occurr_dict[sub_seq] = 1

    # Add special tokens into the token list
    # After making sure that they are have completely exclusive elements
    tokens = set(occurr_dict.keys())
    assert set(special_tokens.values()).isdisjoint(tokens)
    tokens = tokens.union(set(special_tokens.values()))

    token_dict = {}
    for i, c in enumerate(sorted(tokens)):
        if c in occurr_dict:

            prob = occurr_dict[c] / \
                   (total_seq_length - len(protein_seqs) * (len(c) - 1))
            token_dict[c] = (i, prob)
        else:
            token_dict[c] = (i, 1.0)

    if reload_from_disk:
        with open(dict_path, 'w') as f:
            json.dump(token_dict, f, indent=4, separators=(',', ': '))

    return token_dict


def tokenize_protein(
        data_path: str,
        token_dict: dict,
        protein_seqs: iter,
        tokenize_strat: str,
        targets: iter,
        target_token_dict: dict,
        max_seq_length: int = 128,
        reload_from_disk: bool = True):

    special_tokens = SP_TOKENS
    assert tokenize_strat in ['overlapping', 'greedy', 'optimal']

    sos_token = token_dict[special_tokens['SOS']][0]
    eos_token = token_dict[special_tokens['EOS']][0]
    pad_token = token_dict[special_tokens['PAD']][0]

    # Load the tokenized protein sequences if the data file exists
    if os.path.exists(data_path) and reload_from_disk:
        with open(data_path, 'rb') as f:
            tokenized_protein_seqs = pickle.load(f)

    else:
        # Make sure of the data directory
        create_path(os.path.dirname(data_path))

        # Get token length
        token_length = 0
        for k in token_dict.keys():
            if k not in special_tokens.values() and len(k) > token_length:
                token_length = len(k)

        if tokenize_strat == 'overlapping':

            # TODO: overlapping tokenization
            tokenized_protein_seqs = []

        elif tokenize_strat == 'greedy':

            # Parallelized tokenization
            # Parallelization with Joblib actually make things slower due to
            # the large dict size, which exceeds 5 MB.
            #
            # def greedy_tokenize_one_protein_seq(ps):
            #     start_time = time.time()
            #     t = []
            #     curr_index = 0
            #     while curr_index < len(ps):
            #         best_sub_seq = ''
            #         best_prob = 0.
            #         for j in range(1, token_length + 1):
            #             if curr_index + j > len(ps):
            #                 break
            #             sub_seq = ps[curr_index: curr_index + j]
            #             if sub_seq in token_dict:
            #                 prob = token_dict[sub_seq][1] ** (1. / j)
            #                 if prob > best_prob:
            #                     best_sub_seq = sub_seq
            #                     best_prob = prob
            #         assert len(best_sub_seq) != 0
            #         curr_index += len(best_sub_seq)
            #         t.append(token_dict[best_sub_seq][0])
            #     print(time.time() - start_time)
            #     return t
            #
            # num_cores = multiprocessing.cpu_count()
            # print(num_cores)
            # tokenized_protein_seqs = Parallel(n_jobs=num_cores)(
            #     delayed(greedy_tokenize_one_protein_seq)(ps)
            #     for ps in protein_seqs)

            tokenized_protein_seqs = []
            for ps in protein_seqs:

                t = []
                curr_index = 0

                while curr_index < len(ps):

                    best_sub_seq = ''
                    best_prob = 0.

                    for j in range(1, token_length + 1):

                        if curr_index + j > len(ps):
                            break

                        sub_seq = ps[curr_index: curr_index + j]

                        if sub_seq in token_dict:

                            prob = token_dict[sub_seq][1] ** (1. / j)

                            if prob > best_prob:
                                best_sub_seq = sub_seq
                                best_prob = prob

                    assert len(best_sub_seq) != 0
                    curr_index += len(best_sub_seq)
                    t.append(token_dict[best_sub_seq][0])

                # print(time.time() - start_time)
                tokenized_protein_seqs.append(t)

        elif tokenize_strat == 'optimal':

            # TODO: optimal tokenization
            tokenized_protein_seqs = []

        tokenized_protein_seqs = [[sos_token, ] + t + [eos_token, ]
                                  for t in tokenized_protein_seqs]

        # Write the tokenized protein sequences into file (without padding)
        if reload_from_disk:
            with open(data_path, 'wb') as f:
                pickle.dump(tokenized_protein_seqs, f)

    # Take care of padding and only return the strings within given length
    if max_seq_length is None:
        max_seq_length = 0
        for t in tokenized_protein_seqs:
            if len(t) > max_seq_length:
                max_seq_length = len(t)

    ret_protein_seqs = \
        [p for p, t in zip(protein_seqs, tokenized_protein_seqs)
         if len(t) <= max_seq_length]
    ret_tokenized_protein_seqs = \
        [t + [pad_token, ] * (max_seq_length - len(t))
         for t in tokenized_protein_seqs if len(t) <= max_seq_length]

    logger.warning('Keeping %i out of %i (%.2f%%) protein sequences due to '
                   'maximum length limitation.'
                   % (len(ret_protein_seqs), len(protein_seqs),
                      100. * len(ret_protein_seqs)/len(protein_seqs)))

    # Tokenize the target also
    ret_tokenized_targets = \
        [target_token_dict[t] for t, p in zip(targets, tokenized_protein_seqs)
         if len(p) <= max_seq_length]

    return ret_protein_seqs, ret_tokenized_protein_seqs, ret_tokenized_targets


if __name__ == '__main__':

    # Get all the SMILES strings from HIV dataset
    # tasks, (train, valid, test), transformers = \
    #     load_hiv(featurizer='Raw', split='scaffold', reload=True)
    #
    # tokenization_method = 'atom'
    #
    # hiv_token_dict = get_smiles_token_dict(
    #     dict_path='../../data/HIV_%s_token_dict.json' % tokenization_method,
    #     smiles_strings=(list(train.ids) + list(valid.ids) + list(test.ids)),
    #     tokenize_on=tokenization_method)
    #
    # trn_smiles, trn_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/HIV_trn_tokenized_on_%s.pkl'
    #               % tokenization_method,
    #     token_dict=hiv_token_dict,
    #     smiles_strings=train.ids)
    #
    # val_smiles, val_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/HIV_val_tokenized_on_%s.pkl'
    #               % tokenization_method,
    #     token_dict=hiv_token_dict,
    #     smiles_strings=valid.ids)
    #
    # tst_smiles, tst_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/HIV_tst_tokenized_on_%s.pkl'
    #               % tokenization_method,
    #     token_dict=hiv_token_dict,
    #     smiles_strings=test.ids)
    #
    # print(valid.y)
    # print(len(valid.y))
    # print(np.sum(valid.y))
    #
    # print(tst_smiles[0])
    # print(Chem.MolToSmarts(Chem.MolFromSmiles(tst_smiles[0])))
    # print(tst_tokenized_smiles[0])


    # # Get all the SMILES strings from PCBA dataset
    # tasks, (train, valid, test), transformers = \
    #     load_pcba(featurizer='Raw', split='scaffold', reload=True)
    #
    # tokenization_method = 'atom'
    #
    # pcba_token_dict = get_smiles_token_dict(
    #     dict_path='../../data/PCBA_%s_token_dict.json' % tokenization_method,
    #     smiles_strings=(list(train.ids) + list(valid.ids) + list(test.ids)),
    #     tokenize_on='atom', )
    #
    # trn_smiles, trn_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/PCBA_trn_tokenized_on_%s.pkl' %
    # tokenization_method,
    #     token_dict=pcba_token_dict,
    #     smiles_strings=train.ids)
    #
    # val_smiles, val_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/PCBA_val_tokenized_on_%s.pkl' % tokenization_method,
    #     token_dict=pcba_token_dict,
    #     smiles_strings=valid.ids)
    #
    # tst_smiles, tst_tokenized_smiles = tokenize_smiles(
    #     data_path='../../data/PCBA_tst_tokenized_on_%s.pkl' % tokenization_method,
    #     token_dict=pcba_token_dict,
    #     smiles_strings=test.ids)
    #
    # print(valid.y)

    # print(trn_smiles[0])
    #
    # print(len(train.ids))
    # print(len(trn_smiles))
    #
    # print(len(valid.ids))
    # print(len(val_smiles))
    #
    # print(len(test.ids))
    # print(len(tst_smiles))

    # Protein sequence tokenization
    dataframe = pd.read_csv('../../data/coreseed.train.tsv',
                            sep='\t', usecols=['protein', 'function'])

    total_protein_sequences = dataframe['protein']
    total_targets = dataframe['function']
    target_token_dict = dict((f, i) for i, f in
                             enumerate(sorted(set(total_targets))))

    for protein_token_length in [1, 2, 3, 4]:

        protein_token_dict = get_protein_token_dict(
            '../../data/CoreSEED_%i_token_dict.json' % protein_token_length,
            token_length=protein_token_length,
            protein_seqs=total_protein_sequences)

        protein_sequences, tokenized_protein_sequences, tokenized_targets = \
            tokenize_protein(
                '../../data/CoreSEED_trn_tokenized_on_%i.pkl'
                % protein_token_length,
                token_dict=protein_token_dict,
                protein_seqs=total_protein_sequences,
                tokenize_strat='greedy',
                targets=total_targets,
                target_token_dict=target_token_dict,
                max_seq_length=512)

        assert len(protein_sequences) == len(tokenized_protein_sequences)
        assert len(protein_sequences) == len(tokenized_targets)

        print(protein_sequences[0])
        print(tokenized_protein_sequences[0])
        # print(tokenized_targets)
        print(len(target_token_dict))

