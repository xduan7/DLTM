""" 
    File Name:          DLTM/smiles_encoding.py
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


def encode_smiles(
        smiles: iter,
        seq_len: int,

        encoding_on: str = 'atom',

        sos_char: str = '<',
        eos_char: str = '>',
        pad_char: str = ' ',
        mask_char: str = '?',

        data_root: str = '../../data/'):
    """
    WARNING: due to the special way of processing atoms, this encoding
    function does not support SMILES with lowercase atoms (which happens
    when you have aromaticity rings in some format).

    WARNING: also users should make sure that non of these special chars are
    used in SMILES strings. Recommend using the default values.


    :param smiles:
    :param seq_len:
    :param encoding_on:
    :param sos_char:
    :param eos_char:
    :param pad_char:
    :param mask_char:
    :param data_root:
    :return:
    """

    # Pre-processing ##########################################################
    # Sanity check
    assert encoding_on in ['char', 'atom']
    assert len(eos_char) == 1

    create_path(data_root)

    # Get the encoding dictionary #############################################
    # Path (with file name) for encoding dictionary
    dict_path = os.path.join(
        data_root, 'SMILES_%s_encoding_dict.json' % encoding_on)

    # Load the encoding dictionary if it exists already
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            encode_dict = json.load(f)

    # Iterate through all the SMILES and generates encoding dictionary
    else:
        # Create encoding dictionary for SMILES strings based on strategy
        if encoding_on == 'char':

            # Get all the characters in the iterable of SMILES strings
            chars = list(set.union(*[set(s) for s in smiles]).union(
                {sos_char, eos_char, pad_char, mask_char}))

            encode_dict = dict((c, i) for i, c in enumerate(sorted(chars)))

        else:
            # Get all the atoms and symbols in the iterable of SMILES strings
            # Todo: some optimization here?
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
            encode_dict = dict((c, i) for i, c in enumerate(dict_keys))

        # Save encoding dictionary into the path
        with open(dict_path, 'w') as f:
            json.dump(encode_dict, f, indent=4, separators=(',', ': '))

    # Encoding SMILE strings ##################################################
    # encoded_smiles_name
    encoded_smiles_path = os.path.join(
        data_root, 'encoded_SMILES_%s.pkl' % encoding_on)

    # Load the encoded SMILES strings if exist
    if os.path.exists(encoded_smiles_path):
        with open(encoded_smiles_path, 'rb') as f:
            encoded_smiles = pickle.load(f)

    else:
        smiles_ = [(sos_char + s + eos_char) for s in smiles]

        # Encode the SMILES strings differently from strategy
        if encoding_on == 'char':
            encoded_smiles = [[encode_dict[c] for c in s]
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
                        encoded_s.append(encode_dict[s[i: i + 2]])

                    elif not s[i].islower():
                        encoded_s.append(encode_dict[s[i]])

                encoded_smiles.append(encoded_s)

        # Save encoding dictionary into the path
        with open(encoded_smiles_path, 'wb') as f:
            pickle.dump(encoded_smiles, f)

    # Trimming and padding, and one-hot encoding ##############################
    # Only encoding the SMILES strings with length <= max_len
    # Also pad the list of lists with pad_char
    original_smiles = [s for s, e in zip(smiles, encoded_smiles)
                       if len(e) <= seq_len]
    encoded_smiles = [s + [encode_dict[pad_char], ] * (seq_len - len(s))
                      for s in encoded_smiles if len(s) <= seq_len]
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

    return original_smiles, encode_dict, encoded_smiles


if __name__ == '__main__':

    # Get all the SMILES strings
    data_root = '../../data/'
    data_path = os.path.join(data_root, 'dtc.train.filtered.txt')

    import pandas as pd
    df = pd.read_csv(data_path, sep='\t', usecols=['smiles', ])
    smiles = df['smiles'].unique()

    # Encode SMILES strings
    org_smiles, enc_dict, enc_smiles = encode_smiles(smiles, 150, 'atom')
