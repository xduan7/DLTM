""" 
    File Name:          DLTM/params_counting.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/12/18
    Python Version:     3.6.6
    File Description:   

"""
import torch.nn as nn


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

