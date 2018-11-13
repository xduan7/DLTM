""" 
    File Name:          DLTM/encoder.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/8/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn

from networks.modules.embedding import Embedding
from networks.modules.encoder_layer import EncoderLayer
from networks.modules.positional_encoder import PositionalEncoder


class Encoder(nn.Module):

    def __init__(self,
                 dict_size: int,
                 seq_length: int,

                 base_feq: float,
                 emb_scale: float,

                 emb_dim: int,
                 num_layers: int,
                 num_heads: int,
                 ff_mid_dim: int,

                 pe_dropout: float = 0.0,
                 mha_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 enc_dropout: float = 0.0,

                 epsilon: float = 1e-6):

        super().__init__()

        self.__embedding = Embedding(dict_size=dict_size,
                                     emb_dim=emb_dim)

        self.__positional_encoder = \
            PositionalEncoder(seq_length=seq_length,
                              emb_dim=emb_dim,
                              emb_scale=emb_scale,
                              dropout=pe_dropout,
                              base_feq=base_feq)

        self.__encoder_layers = nn.ModuleList(
            [EncoderLayer(emb_dim=emb_dim,
                          num_heads=num_heads,
                          ff_mid_dim=ff_mid_dim,
                          mha_dropout=mha_dropout,
                          ff_dropout=ff_dropout,
                          enc_dropout=enc_dropout,
                          epsilon=epsilon) for _ in range(num_layers)])

        self.__output_norm = nn.LayerNorm(normalized_shape=emb_dim,
                                          eps=epsilon)

    def forward(self, src_indexed_sentence, src_mask):

        h = self.__positional_encoder(self.__embedding(src_indexed_sentence))
        for encoder_layer in self.__encoder_layers:
            h = encoder_layer(h, src_mask)

        return self.__output_norm(h)
