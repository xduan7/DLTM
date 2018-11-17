""" 
    File Name:          DLTM/dense_clf.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/12/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn
import torch.nn.functional as F

from networks.modules.embedding import Embedding
from networks.modules.positional_encoder import PositionalEncoder


class DenseClf(nn.Module):

    def __int__(self,
                dict_size: int,
                seq_length: int,

                base_feq: float,
                emb_scale: float,

                emb_dim: int,
                intermediate_dim: int,
                num_intermediate_layers: int,

                pe_dropout: float = 0.1):

        super().__init__()
        assert num_intermediate_layers > 0

        # Use word embedding and positional encoding to make it a fair
        # comparison with transformer (encoder)
        self.__embedding = Embedding(dict_size=dict_size,
                                     emb_dim=emb_dim)

        self.__positional_encoder = \
            PositionalEncoder(seq_length=seq_length,
                              emb_dim=emb_dim,
                              emb_scale=emb_scale,
                              dropout=pe_dropout,
                              base_feq=base_feq)

        input_layer = nn.Linear(seq_length * emb_dim, intermediate_dim)
        output_layer = nn.Linear(intermediate_dim, dict_size)

        self.__layers = [input_layer, nn.ReLU()] + \
            [nn.Linear(intermediate_dim, intermediate_dim)
             for _ in range(num_intermediate_layers - 1)] + \
            [output_layer, ]

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, indexed_sentences, mask=None):

        batch_size = indexed_sentences.size(0)

        # x is of size (batch_size, seq_length, emb_dim)
        x = self.__positional_encoder(self.__embedding(indexed_sentences))

        # Apply mask if available
        if mask is not None:
            print(mask.size)
            print(mask.unsqueeze(-1).unsqueeze(-2).size)
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-2) == 0, -1e9)

        x = x.view(batch_size, -1)

        for layer in self.__layers:
            x = layer(x, mask)

        return F.log_softmax(x, dim=-1)
