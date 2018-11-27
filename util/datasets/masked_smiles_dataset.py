""" 
    File Name:          DLTM/masked_smiles_dataset.py
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
from util.data_processing.tokenization import \
    get_smiles_token_dict, tokenize_smiles

logger = logging.getLogger(__name__)


class MaskedSMILESDataset(data.Dataset):

    def __init__(self,
                 smiles: str,
                 training: bool,
                 rand_state: int = 0,

                 tokenize_on: str = 'atom',
                 max_seq_length: int = 128,
                 val_ratio: float = 0.1):

        self.__rand_state = rand_state

        # Tokenize the SMILES strings (with tokenization dict)
        token_dict = get_smiles_token_dict(dict_path=token_dict_path,
                                           tokenize_on=tokenize_on,
                                           smiles_strings=smiles)

        smiles, tokenized_smiles = \
            tokenize_smiles(data_path=tokenized_data_path,
                            token_dict=token_dict,
                            smiles_strings=smiles,
                            max_seq_length=max_seq_length)

        # Mask encoded smile strings (1 mask per string)
        masked_values, masked = \
            mask_sentences(mask=token_dict['<UNK>'],
                           sentences=smiles,
                           tokenized_sentences=tokenized_smiles,
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
            (self.__data != self.token_dict['<PAD>'])).astype(np.int64)

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

    dataset = MaskedSMILESDataset(data_name='DTC',
                                  data_root='../../data/',
                                  data_file_name='dtc.train.filtered.txt',
                                  training=True)
    m, d, t = dataset[-1]

    print(np.multiply(m, d))
