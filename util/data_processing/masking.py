""" 
    File Name:          DLTM/masking.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/8/18
    Python Version:     3.6.6
    File Description:   

"""
import copy
import random


def mask(original_str: iter,
         encoded_str: iter,
         mask_value: int,
         rand_state: int = 0):

    random.seed(rand_state)
    masked_encoded_str = copy.deepcopy(encoded_str)
    masked_indices = []
    masked_values = []

    for o, m in zip(original_str, masked_encoded_str):
        index = random.randint(1, len(o))
        masked_indices.append(index)
        masked_values.append(m[index])
        m[index] = mask_value

    return masked_indices, masked_values, masked_encoded_str
