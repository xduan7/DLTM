""" 
    File Name:          DLTM/decoder.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/8/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn

from networks.transformer.transformer import Embedding
from networks.transformer.transformer import DecoderLayer
from networks.transformer.transformer import PositionalEncoder


class Decoder(nn.Module):
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

        self.__decoder_layers = nn.ModuleList(
            [DecoderLayer(emb_dim=emb_dim,
                          num_heads=num_heads,
                          ff_mid_dim=ff_mid_dim,
                          mha_dropout=mha_dropout,
                          ff_dropout=ff_dropout,
                          enc_dropout=enc_dropout,
                          epsilon=epsilon) for _ in range(num_layers)])

        self.__output_norm = nn.LayerNorm(normalized_shape=emb_dim,
                                          eps=epsilon)

    def forward(self, trg_indexed_sentence, encoder_output,
                src_mask, trg_mask):

        h = self.__positional_encoder(self.__embedding(trg_indexed_sentence))
        for decoder_layer in self.__decoder_layers:
            h = decoder_layer(h, encoder_output, src_mask, trg_mask)
        return self.__output_norm(h)
