""" 
    File Name:          DLTM/encoder_clf.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/12/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn
import torch.nn.functional as F

from networks.modules.encoder import Encoder


class EncoderClf(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 output_module: nn.Module):

        super().__init__()

        self.__encoder = encoder
        self.__output_module = output_module

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, indexed_sentences, mask):

        batch_size = mask.size(0)

        # Make sure that the mask match the embedded structure
        encoder_output = self.__encoder(indexed_sentences, mask.unsqueeze(-2))

        return self.__output_module(encoder_output.view(batch_size, -1))
