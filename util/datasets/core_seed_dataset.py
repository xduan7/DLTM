""" 
    File Name:          DLTM/core_seed_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/15/18
    Python Version:     3.6.6
    File Description:   

"""
import json
import os
import logging
import numpy as np
import pandas as pd
import torch.utils.data as data

logger = logging.getLogger(__name__)


class CoreSEEDDataset(data.Dataset):

    def __init__(self,
                 data_root: str,
                 training: bool,
                 rand_state: int = 0,

                 max_seq_len: int = 1024,
                 sos_char: str = '<',
                 eos_char: str = '>',
                 pad_char: str = ' '):

        self.__rand_state = rand_state

        # Should probably merge training and testing set anyway
        # And get all the amino acids token dict as well as all the protein
        # functions

        # Check if the token dict and class encoding exists
        prt_token_dict_name = 'CoreSEED_protein_token_dict.json'
        prt_token_dict_path = os.path.join(data_root,
                                           prt_token_dict_name)

        fcn_dict_name = 'CoreSEED_function_dict.json'
        fcn_dict_path = os.path.join(data_root, fcn_dict_name)

        if os.path.exists(prt_token_dict_path) and \
                os.path.exists(fcn_dict_path):

            with open(prt_token_dict_path, 'r') as f:
                prt_token_dict = json.load(f)

            with open(fcn_dict_path, 'r') as f:
                fcn_dict = json.load(f)

            file_name = 'coreseed.train.tsv' if training \
                else 'coreseed.test.tsv'
            data_path = os.path.join(data_root, file_name)
            dataframe = pd.read_csv(data_path, sep='\t',
                                    usecols=['function', 'protein'])

        else:
            trn_dataframe = pd.read_csv(
                os.path.join(data_root, 'coreseed.train.tsv'),
                sep='\t', usecols=['function', 'protein'])
            val_dataframe = pd.read_csv(
                os.path.join(data_root, 'coreseed.test.tsv'),
                sep='\t', usecols=['function', 'protein'])

            merged_dataframe = pd.concat([trn_dataframe, val_dataframe],
                                         ignore_index=True)

            # Get all the protein sequences and tokenize
            prt_array = merged_dataframe['protein'].unique()

            sp_chars = {sos_char, eos_char, pad_char}
            prt_token_list = list(
                set.union(*[set(p) for p in prt_array]).union(sp_chars))

            print(prt_token_list)
            print(len(prt_token_list))

            prt_token_dict = dict((t, i) for i, t in enumerate(
                sorted(prt_token_list)))

            # Save the protein token dict
            with open(prt_token_dict_path, 'w') as f:
                json.dump(prt_token_dict, f, indent=4, separators=(',', ': '))

            # Get all the functions and tokenize
            fcn_array = merged_dataframe['function'].unique()
            fcn_dict = dict((f, i) for i, f in enumerate(sorted(fcn_array)))

            with open(fcn_dict_path, 'w') as f:
                json.dump(fcn_dict, f, indent=4, separators=(',', ': '))

            dataframe = trn_dataframe if training else val_dataframe

        # Tokenize all the proteins in dataframe
        # Tokenize all the corresponding protein functions
        self.__protein = dataframe['protein'].tolist()
        self.__indexed_protein = \
            [[prt_token_dict[c] for c in (sos_char + p + eos_char)]
             for p in self.__protein]

        self.__function = dataframe['function'].tolist()
        self.__indexed_function = [fcn_dict[f] for f in self.__function]

        self.__indexed_function = \
            [f for f, p in zip(self.__indexed_function, self.__indexed_protein)
             if len(p) <= max_seq_len]
        self.__indexed_protein = \
            [p + [prt_token_dict[pad_char], ] * (max_seq_len - len(p))
             for p in self.__indexed_protein if len(p) <= max_seq_len]

        assert len(self.__indexed_protein) == len(self.__indexed_function)

        self.__len = len(self.__indexed_protein)
        self.fcn_dict = fcn_dict
        self.prt_token_dict = prt_token_dict

        self.__indexed_protein = \
            np.array(self.__indexed_protein).astype(np.int64)
        self.__indexed_function = \
            np.array(self.__indexed_function).astype(np.int64)

        self.__padding_mask = \
            np.array((self.__indexed_protein
                      != self.prt_token_dict[pad_char])).astype(np.int64)

    def __len__(self):
        """len(dataset)

        Returns:
            int: length of this dataset
        """
        return self.__len

    def __getitem__(self, index):
        """input_mask, protein_seq, function = dataset[0]

        This function returns the padded, indexed (encoded) protein
        sequences and the its target function.

        Note that every element is of type np.int64, which feeds into
        PyTorch embedding layer.

        Args:
            index (int): index of the data.

        Returns:
            (
                np.array: padding mask (0 means padding)
                np.array: indexed padded protein sequence
                np.array: indexed protein function
            )
        """
        return self.__padding_mask[index], \
               self.__indexed_protein[index], \
               self.__indexed_function[index]


if __name__ == '__main__':

    dataset = CoreSEEDDataset(data_root='../../data/',
                              max_seq_len=1024,
                              training=False)

    print(dataset[0])
    print(dataset[1])

    print(len(dataset.fcn_dict))
