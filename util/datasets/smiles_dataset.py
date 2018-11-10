""" 
    File Name:          DLTM/smiles_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/5/18
    Python Version:     3.6.6
    File Description:   

"""
import os
import logging
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from util.data_processing.masking import mask
from util.data_processing.tokenization import tokenize_smiles


logger = logging.getLogger(__name__)


class SMILESDataset(data.Dataset):

    def __init__(
            self,

            data_root: str,
            training: bool,
            rand_state: int = 0,

            max_seq_len: int = 150,
            tokenize_on: str = 'atom',
            sos_char: str = '<',
            eos_char: str = '>',
            pad_char: str = ' ',
            mask_char: str = '?',

            val_ratio: float = 0.1):

        self.__rand_state = rand_state

        # Get the SMILES strings from dataframe
        data_path = os.path.join(data_root,
                                 'dtc.train.filtered.txt')
        smiles = pd.read_csv(data_path, sep='\t')['smiles'].unique()

        # Encode the SMILES strings
        smiles, encode_dict, encoded = \
            tokenize_smiles(smiles=smiles,
                            max_seq_len=max_seq_len,
                            tokenize_on=tokenize_on,
                            sos_char=sos_char,
                            eos_char=eos_char,
                            pad_char=pad_char,
                            mask_char=mask_char,
                            data_root=data_root)

        # Mask encoded smile strings (1 mask per string)
        masked_indices, masked_values, masked_encoded = \
            mask(original_str=smiles,
                 encoded_str=encoded,
                 mask_value=encode_dict[mask_char],
                 rand_state=rand_state)

        # Train/test split
        trn_encoded, val_encoded, \
            trn_masked_encoded, val_masked_encoded, \
            trn_masked_indices, val_masked_indices, \
            trn_masked_values, val_masked_values = \
            train_test_split(encoded,
                             masked_encoded,
                             masked_indices,
                             masked_values,
                             test_size=val_ratio,
                             random_state=self.__rand_state,
                             shuffle=True)

        self.encode_dict = encode_dict
        if training:
            self.__encoded = trn_encoded
            self.__masked_encoded = trn_masked_encoded
            self.__masked_indices = trn_masked_indices
            self.__masked_values = trn_masked_values
        else:
            self.__encoded = val_encoded
            self.__masked_encoded = val_masked_encoded
            self.__masked_indices = val_masked_indices
            self.__masked_values = val_masked_values

        self.__encoded = np.array(self.__encoded).astype(np.int64)
        self.__masked_encoded = \
            np.array(self.__masked_encoded).astype(np.int64)
        self.__pad_mask = np.array(
            (self.__encoded != self.encode_dict[pad_char])).astype(np.int64)
        self.__masked_values = np.array(self.__masked_values).astype(np.int64)

        self.__len = len(self.__encoded)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.__pad_mask[index], self.__masked_encoded[index], \
               self.__masked_values[index]


if __name__ == '__main__':
    dataset = SMILESDataset('../../data/', True, rand_state=0)
    print(dataset[-1])
