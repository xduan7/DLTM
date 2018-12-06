""" 
    File Name:          DLTM/pcba_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               12/5/18
    Python Version:     3.6.6
    File Description:   

"""

import logging
import numpy as np
import torch.utils.data as data
from deepchem.molnet import load_pcba

from util.data_processing.tokenization import get_smiles_token_dict, \
    tokenize_smiles

logger = logging.getLogger(__name__)


class PCBADataset(data.Dataset):

    def __init__(self,
                 token_dict_path: str,
                 tokenized_data_path: str,
                 dataset_usage: str,
                 rand_state: int = 0,

                 tokenize_on: str = 'atom',
                 max_seq_length: int = 256):

        assert dataset_usage in ['training', 'validation', 'test']
        self.__rand_state = rand_state

        tasks, (trn, val, tst), transformers = \
            load_pcba(featurizer='Raw', split='random', reload=True)

        # seq_length = 0
        # for i in (list(trn.ids) + list(val.ids) + list(tst.ids)):
        #     if len(i) > seq_length:
        #         seq_length = len(i)
        #
        # print(seq_length)

        token_dict = get_smiles_token_dict(
            dict_path=token_dict_path,
            smiles_strings=(list(trn.ids) + list(val.ids) + list(tst.ids)),
            tokenize_on=tokenize_on)

        if dataset_usage == 'training':
            smiles = trn.ids
            targets = trn.y
        elif dataset_usage == 'validation':
            smiles = val.ids
            targets = val.y
        else:
            smiles = tst.ids
            targets = tst.y

        targets = np.argmax(targets, axis=1)

        self.__smiles, tokenized_smiles, targets = tokenize_smiles(
            data_path=tokenized_data_path,
            token_dict=token_dict,
            smiles_strings=smiles,
            targets=targets,
            max_seq_length=max_seq_length)

        # Create a mask for padding characters in each SMILES string
        self.token_dict = token_dict
        self.__padding_mask = \
            np.array((np.array(tokenized_smiles)
                      != self.token_dict['<PAD>'])).astype(np.int64)

        # Convert the data and target type to int64 to work with PyTorch
        self.__data = np.array(tokenized_smiles).astype(np.int64)
        self.__targets = np.array(targets).astype(np.int64)

        self.__len = len(self.__data)

    def __len__(self):

        return self.__len

    def __getitem__(self, index):
        return self.__padding_mask[index], \
               self.__data[index], self.__targets[index]


if __name__ == '__main__':
    dataset = PCBADataset(
        token_dict_path='../../data/PCBA_atom_token_dict.json',
        tokenized_data_path='../../data/PCBA_trn_tokenized_on_atom.pkl',
        dataset_usage='validation')

    m, d, t = dataset[-2]
    print(np.multiply(m, d))

    print(len(dataset))

    print(m)
    print(d)
    print(t)
