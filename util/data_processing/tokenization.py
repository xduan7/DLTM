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


def tokenize_smiles(
        smiles: iter,
        max_seq_len: int,
        tokenize_on: str = 'atom',

        sos_char: str = '<',
        eos_char: str = '>',
        pad_char: str = ' ',
        mask_char: str = '?',

        data_root: str = '../../data/'):
    """smiles, tokens, indexed = tokenize_smiles(smiles_strings, 120)

    This function tokenizes a series of SMILES strings based either on
    characters or atoms/bonds, with the special characters specified.

    WARNING: due to the special way of processing atoms, this encoding
    function does not support SMILES with lowercase atoms (which happens
    when you have aromaticity rings in some format).

    WARNING: also users should make sure that non of these special chars are
    used in SMILES strings. Recommend using the default values.

    Args:
        smiles (iter): a iterable structure of SMILES strings (type: str)
        max_seq_len (int): the maximum length of the SMILES strings allowed
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
            list: list of SMILES strings with valid length but no padding
            dict: tokenization dictionary for SMILES -> numbers
            list: list of indexed (numeric) SMILES strings with padding
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
        data_root, 'SMILES_%s_token_dict.json' % tokenize_on)

    # Load the tokenization dictionary if it exists already
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            token_dict = json.load(f)

    # Iterate through all the SMILES and generates tokenization dictionary
    else:

        # Create encoding dictionary for SMILES strings based on strategy
        # Note that all the token are sorted before putting into dictionary
        if tokenize_on == 'char':

            # Collect all the characters in SMILES strings
            # Make sure that all the special characters are included
            chars = list(set.union(*[set(s) for s in smiles]).union(
                # Make sure that all the special characters are included
                {sos_char, eos_char, pad_char, mask_char}))

            token_dict = dict((c, i) for i, c in enumerate(sorted(chars)))

        else:
            # Collect all the atoms and bonds in SMILES strings
            # Make sure that all the special characters are included
            dict_keys = {sos_char, eos_char, pad_char, mask_char}

            for s in smiles:
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

    # Index SMILE strings #####################################################
    indexed_smiles_path = os.path.join(
        data_root, 'indexed_SMILES_%s.pkl' % tokenize_on)

    # Load the indexed SMILES strings if exist
    if os.path.exists(indexed_smiles_path):
        with open(indexed_smiles_path, 'rb') as f:
            indexed_smiles = pickle.load(f)

    else:
        smiles_ = [(sos_char + s + eos_char) for s in smiles]

        # Index (numeric) the SMILES strings differently from strategy
        if tokenize_on == 'char':
            indexed_smiles = [[token_dict[c] for c in s]
                              for s in smiles_]

        else:
            # Get all the atoms and symbols in the iterable of SMILES strings
            # Todo: some optimization here?
            indexed_smiles = []

            for s in smiles_:

                indexed_s = []

                for i in range(len(s)):

                    # All lower-case letters are the last letter of an atom
                    if i < len(s) - 1 and \
                            (s[i].isupper() and s[i + 1].islower()):
                        indexed_s.append(token_dict[s[i: i + 2]])

                    elif not s[i].islower():
                        indexed_s.append(token_dict[s[i]])

                indexed_smiles.append(indexed_s)

        # Save indexed (numeric) SMILES strings into the path
        with open(indexed_smiles_path, 'wb') as f:
            pickle.dump(indexed_smiles, f)

    # Trimming and padding ####################################################
    # Only encoding the SMILES strings with length <= max_len
    # Also pad the list of lists with pad_char
    smiles = [s for s, i in zip(smiles, indexed_smiles)
              if len(i) <= max_seq_len]
    indexed_smiles = [s + [token_dict[pad_char], ] * (max_seq_len - len(s))
                      for s in indexed_smiles if len(s) <= max_seq_len]

    return smiles, token_dict, indexed_smiles


if __name__ == '__main__':

    # Get all the SMILES strings
    d_root = '../../data/'
    d_path = os.path.join(d_root, 'dtc.train.filtered.txt')

    import pandas as pd
    df = pd.read_csv(d_path, sep='\t', usecols=['smiles', ])

    # Index SMILES strings
    list_smiles, token_dict, list_indexed_smiles = \
        tokenize_smiles(df['smiles'].unique(), 128, 'atom')

    print(list_smiles[0])
    print(list_indexed_smiles[0])
