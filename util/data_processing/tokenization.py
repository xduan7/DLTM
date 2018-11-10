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
    """
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
    encoded_smiles_path = os.path.join(
        data_root, 'encoded_SMILES_%s.pkl' % tokenize_on)

    # Load the encoded SMILES strings if exist
    if os.path.exists(encoded_smiles_path):
        with open(encoded_smiles_path, 'rb') as f:
            encoded_smiles = pickle.load(f)

    else:
        smiles_ = [(sos_char + s + eos_char) for s in smiles]

        # Encode the SMILES strings differently from strategy
        if tokenize_on == 'char':
            encoded_smiles = [[token_dict[c] for c in s]
                              for s in smiles_]

        else:
            # Get all the atoms and symbols in the iterable of SMILES strings
            # Todo: some optimization here?
            encoded_smiles = []

            for s in smiles_:

                encoded_s = []

                for i in range(len(s)):

                    # All lower-case letters are the last letter of an atom
                    if i < len(s) - 1 and \
                            (s[i].isupper() and s[i + 1].islower()):
                        encoded_s.append(token_dict[s[i: i + 2]])

                    elif not s[i].islower():
                        encoded_s.append(token_dict[s[i]])

                encoded_smiles.append(encoded_s)

        # Save encoding dictionary into the path
        with open(encoded_smiles_path, 'wb') as f:
            pickle.dump(encoded_smiles, f)

    # Trimming and padding, and one-hot encoding ##############################
    # Only encoding the SMILES strings with length <= max_len
    # Also pad the list of lists with pad_char
    original_smiles = [s for s, e in zip(smiles, encoded_smiles)
                       if len(e) <= max_seq_len]
    encoded_smiles = [s + [token_dict[pad_char], ] * (max_seq_len - len(s))
                      for s in encoded_smiles if len(s) <= max_seq_len]
    encoded_smiles = np.array(encoded_smiles).astype(np.uint8)

    # Now we have a np array of encoded SMILES with length = max_len and ended
    # with encoded EOS. Suppose the shape is [n ,max_len]

    # One-hot encoding [n ,max_len] -> [n ,max_len, len(encode_dict)]
    # if output_format == 'one_hot':
    #
    #     one_hot_encoded_smiles = np.zeros(
    #         (len(encoded_smiles), max_len, len(encode_dict)), dtype=np.uint8)
    #
    #     for i, s in enumerate(encoded_smiles):
    #         for j, c in enumerate(s):
    #             one_hot_encoded_smiles[i, j, c] = 1
    #     encoded_smiles = one_hot_encoded_smiles

    return original_smiles, token_dict, encoded_smiles


if __name__ == '__main__':

    # Get all the SMILES strings
    d_root = '../../data/'
    d_path = os.path.join(d_root, 'dtc.train.filtered.txt')

    import pandas as pd
    df = pd.read_csv(d_path, sep='\t', usecols=['smiles', ])
    smiles = df['smiles'].unique()

    # Encode SMILES strings
    org_smiles, enc_dict, enc_smiles = \
        tokenize_smiles(smiles, 150, 'atom', d_root)
