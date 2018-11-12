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

from util.data_processing.sentence_masking import mask_sentences
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
        """



        Args:
            data_root:
            training:
            rand_state:
            max_seq_len:
            tokenize_on:
            sos_char:
            eos_char:
            pad_char:
            mask_char:
            val_ratio:
        """

        self.__rand_state = rand_state

        # Get the SMILES strings from file
        data_path = os.path.join(data_root, 'dtc.train.filtered.txt')
        smiles = pd.read_csv(data_path, sep='\t')['smiles'].unique()

        # Index the SMILES strings (making it numeric with tokenization dict)
        smiles, token_dict, indexed = \
            tokenize_smiles(smiles=smiles,
                            max_seq_len=max_seq_len,
                            tokenize_on=tokenize_on,
                            sos_char=sos_char,
                            eos_char=eos_char,
                            pad_char=pad_char,
                            mask_char=mask_char,
                            data_root=data_root)

        # This might save some ram and processing time
        indexed = np.array(indexed).astype(np.uint8)

        # Mask encoded smile strings (1 mask per string)
        masked_values, masked = \
            mask_sentences(mask=token_dict[mask_char],
                           sentences=smiles,
                           indexed_sentences=indexed,
                           rand_state=rand_state)

        # Train/test split the masked SMILES strings and targets
        trn_data, val_data, trn_target, val_target = \
            train_test_split(masked,
                             masked_values,
                             test_size=val_ratio,
                             random_state=self.__rand_state,
                             shuffle=True)

        # Save everything useful into private variables
        self.token_dict = token_dict
        if training:
            self.__data = trn_data
            self.__target = trn_target
        else:
            self.__data = val_data
            self.__target = val_target

        # Create a mask for padding characters in each SMILES string
        self.__padding_mask = np.array(
            (self.__data != self.token_dict[pad_char])).astype(np.int64)

        # Convert the data and target type to int64 to work with PyTorch
        self.__data = np.array(self.__data).astype(np.int64)
        self.__target = np.array(self.__target).astype(np.int64)

        self.__len = len(self.__data)

    def __len__(self):
        """len(dataset)

        Returns:
            int: length of this dataset
        """
        return self.__len

    def __getitem__(self, index):
        """mask, data, target = dataset[0]

        This function returns the padding mask, data, and target
        corresponding to a certain index.

        Note that every element is of type np.int64, which feeds into
        PyTorch embedding layer.

        Args:
            index (int): index of the data.

        Returns:
            (
                np.array: padding mask (0 means padding)
                np.array: numeric SMILES string with all the special
                    characters and the masked element to be predicted
                np.array: actual value under the mask for prediction target
            )
        """
        return self.__padding_mask[index], \
               self.__data[index], self.__target[index]


if __name__ == '__main__':

    dataset = SMILESDataset('../../data/', True, rand_state=0)
    m, d, t = dataset[-1]

    print(np.multiply(m, d))
