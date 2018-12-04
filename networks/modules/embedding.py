""" 
    File Name:          DLTM/embedding.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/7/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self,
                 dict_size: int,
                 emb_dim: int):

        super().__init__()
        self.__embedding = nn.Embedding(dict_size,
                                        emb_dim)

    def forward(self, x):

        # Input size: (batch_size, seq_length)
        # Output size: (batch_size, seq_length, emb_dim)

        return self.__embedding(x)
