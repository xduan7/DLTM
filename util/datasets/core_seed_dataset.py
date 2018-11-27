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

from util.data_processing.tokenization import get_protein_token_dict, \
    tokenize_protein, SP_TOKENS

logger = logging.getLogger(__name__)


class CoreSEEDDataset(data.Dataset):

    def __init__(self,
                 training: bool,
                 data_root: str,
                 token_length: int,
                 rand_state: int = 0,
                 max_seq_length: int = 1024):

        self.__rand_state = rand_state

        # Check if the token dict and class encoding exists

        function_token_dict_name = 'CoreSEED_function_token_dict.json'
        function_token_dict_path = \
            os.path.join(data_root, function_token_dict_name)

        if os.path.exists(function_token_dict_path):

            with open(function_token_dict_path, 'r') as f:
                function_token_dict = json.load(f)

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

            # Get all the functions and tokenize
            functions = merged_dataframe['function'].unique()
            function_token_dict = dict((f, i) for i, f in
                                       enumerate(sorted(functions)))

            with open(function_token_dict_path, 'w') as f:
                json.dump(function_token_dict, f,
                          indent=4, separators=(',', ': '))

            dataframe = trn_dataframe if training else val_dataframe

        protein_token_dict_path = \
            os.path.join(data_root,
                         'CoreSEED_%i_token_dict.json' % token_length)

        protein_token_dict = get_protein_token_dict(
            dict_path=protein_token_dict_path,
            token_length=token_length,
            protein_seqs=dataframe['protein'])

        tokenized_protein_file_name = \
            'CoreSEED_trn_tokenized_on_%i.pkl' % token_length if training \
            else 'CoreSEED_val_tokenized_on_%i.pkl' % token_length

        tokenized_protein_path = \
            os.path.join(data_root, tokenized_protein_file_name)

        protein_sequences, tokenized_protein_sequences, tokenized_targets = \
            tokenize_protein(data_path=tokenized_protein_path,
                             token_dict=protein_token_dict,
                             protein_seqs=dataframe['protein'],
                             tokenize_strat='greedy',
                             targets=dataframe['function'],
                             target_token_dict=function_token_dict,
                             max_seq_length=max_seq_length)

        assert len(protein_sequences) == len(tokenized_protein_sequences)
        assert len(protein_sequences) == len(tokenized_targets)

        self.__len = len(protein_sequences)
        self.protein_token_dict = protein_token_dict
        self.function_token_dict = function_token_dict

        self.__tokenized_protein_sequences = \
            np.array(tokenized_protein_sequences).astype(np.int64)
        self.__tokenized_targets = \
            np.array(tokenized_targets).astype(np.int64)

        pad_token = protein_token_dict[SP_TOKENS['PAD']][0]
        self.__padding_mask = np.array(
            (self.__tokenized_protein_sequences != pad_token)).astype(np.int64)

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
               self.__tokenized_protein_sequences[index], \
               self.__tokenized_targets[index]


if __name__ == '__main__':

    dataset = CoreSEEDDataset(training=True,
                              data_root='../../data/',
                              token_length=1,
                              max_seq_length=512)

    print(dataset[0])
    print(dataset[1])

    print(len(dataset.function_token_dict))
