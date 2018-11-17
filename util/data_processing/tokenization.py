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
import numpy as np

from util.misc.path_creating import create_path


def tokenize(
        data_name: str,
        sentences: iter,
        max_seq_len: int,
        tokenize_on: str = 'atom',

        sos_char: str = '<',
        eos_char: str = '>',
        pad_char: str = ' ',
        mask_char: str = '?',

        data_root: str = '../../data/'):
    """sentences, token_dict, indexed_sentences = \
        tokenize('name', strings, 120)

    This function tokenizes a series of strings based either on
    characters or atoms/bonds, with the special characters specified.

    WARNING: due to the special way of processing atoms, this encoding
    function does not support SMILES with lowercase atoms (which happens
    when you have aromaticity rings in some format).

    WARNING: also users should make sure that non of these special chars are
    used in sentences. Recommend using the default values.

    Args:
        data_name (str): name of this dataset
        sentences (iter): a iterable structure of sentences (type: str)
        max_seq_len (int): the maximum length of the sentences allowed
            for tokenization (including special chars like SOS and EOS)
        tokenize_on (str): tokenization strategy. Choose between 'atom'
            (tokenize on atoms/bonds)and 'char' (tokenize on characters)

        sos_char (str): string/character indicating the start of sentence
        eos_char (str): string/character indicating the end of sentence
        pad_char (str): string/character indicating sentence padding
        mask_char (str): string/character indicating word masking

        data_root (str): path to data folder

    Returns:
        (
            list: list of sentences with valid length but no padding
            dict: tokenization dictionary for words -> numbers
            list: list of indexed (numeric) sentences with padding
                and valid length
        )
    """

    # Pre-processing ##########################################################
    # Make sure of the tokenize strategy and existence of data path
    assert tokenize_on in ['char', 'atom']
    create_path(data_root)

    # Get the tokenization dictionary #########################################
    # Path (with file name) for tokenization dictionary
    dict_path = os.path.join(
        data_root, '%s_%s_token_dict.json' % (data_name, tokenize_on))

    # Load the tokenization dictionary if it exists already
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            token_dict = json.load(f)

    # Iterate through all the sentences and generates tokenization dictionary
    else:

        # Create encoding dictionary for words based on strategy
        # Note that all the token are sorted before putting into dictionary
        if tokenize_on == 'char':

            if mask_char:
                chars = list(set.union(*[set(s) for s in sentences]).union(
                    {sos_char, eos_char, pad_char, mask_char}))
            else:
                chars = list(set.union(*[set(s) for s in sentences]).union(
                    {sos_char, eos_char, pad_char}))

            token_dict = dict((c, i) for i, c in enumerate(sorted(chars)))

        else:
            if mask_char:
                dict_keys = {sos_char, eos_char, pad_char, mask_char}
            else:
                dict_keys = {sos_char, eos_char, pad_char, }

            for s in sentences:
                for i in range(len(s)):

                    # All lower-case letters are the last letter of an atom
                    if i < len(s) - 1 and \
                            (s[i].isupper() and s[i + 1].islower()):
                        dict_keys.add(s[i: i + 2])

                    elif not s[i].islower():
                        dict_keys.add(s[i])

            dict_keys = sorted(list(dict_keys))
            token_dict = dict((c, i) for i, c in enumerate(dict_keys))

        # Save tokenization dictionary into the path
        with open(dict_path, 'w') as f:
            json.dump(token_dict, f, indent=4, separators=(',', ': '))

    # Index sentences #########################################################
    indexed_sentences_file_path = os.path.join(
        data_root, 'indexed_%s_%s.pkl' % (data_name, tokenize_on))

    # Load the indexed SMILES strings if exist
    if os.path.exists(indexed_sentences_file_path):
        with open(indexed_sentences_file_path, 'rb') as f:
            indexed_sentences = pickle.load(f)

    else:
        sentences_ = [(sos_char + s + eos_char) for s in sentences]

        # Index (numeric) the sentences differently from strategy
        if tokenize_on == 'char':
            indexed_sentences = [[token_dict[c] for c in s]
                                 for s in sentences_]

        else:
            # Get all the atoms and symbols in the iterable of SMILES strings
            # Todo: some optimization here?
            indexed_sentences = []

            for s in sentences_:

                indexed_s = []

                for i in range(len(s)):

                    # All lower-case letters are the last letter of an atom
                    if i < len(s) - 1 and \
                            (s[i].isupper() and s[i + 1].islower()):
                        indexed_s.append(token_dict[s[i: i + 2]])

                    elif not s[i].islower():
                        indexed_s.append(token_dict[s[i]])

                indexed_sentences.append(indexed_s)

        # Save indexed (numeric) sentences into the path
        with open(indexed_sentences_file_path, 'wb') as f:
            pickle.dump(indexed_sentences, f)

    # Trimming and padding ####################################################
    # Only encoding the SMILES strings with length <= max_len
    # Also pad the list of lists with pad_char
    sentences = [s for s, i in zip(sentences, indexed_sentences)
                 if len(i) <= max_seq_len]
    indexed_sentences = [s + [token_dict[pad_char], ] * (max_seq_len - len(s))
                         for s in indexed_sentences if len(s) <= max_seq_len]

    return sentences, token_dict, indexed_sentences


if __name__ == '__main__':

    # Get all the SMILES strings
    d_root = '../../data/'
    d_path = os.path.join(d_root, 'dtc.train.filtered.txt')

    import pandas as pd
    df = pd.read_csv(d_path, sep='\t', usecols=['smiles', ])

    # Index SMILES strings
    list_smiles, token_dict, list_indexed_smiles = \
        tokenize(data_name='DTC_SMILES',
                 sentences=df['smiles'].unique(),
                 max_seq_len=128,
                 tokenize_on='char')

    print(list_smiles[0])
    print(list_indexed_smiles[0])
