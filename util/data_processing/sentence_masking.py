""" 
    File Name:          DLTM/sentence_masking.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/8/18
    Python Version:     3.6.6
    File Description:   

"""
import copy
import random


def mask_sentences(mask,
                   sentences: iter,
                   indexed_sentences: iter,
                   rand_state: int = 0):
    """masked_indexed_sentences = mask_sentences(
        0, sentences, indexed_sentences)

    This function takes an iterable structure of strings and mask one part
    of the indexed and padded version of the string randomly.

    For example, suppose that padding = 0, mask = 99, sentences = [[
    'some_sentence']], indexed_sentences = [[1, 2, 3, 4, 5, 1, ..., 4,
    0, 0, ..., 0]]. The function is now going to pick an index within the
    range of the length of the actual sentence (not the indexed one),
    and flip the number at the index of the indexed sentences to mask value,
    in this case, 99. Suppose that the function picked index 0 for the
    sentence #0, then the returned list will probably look like [[99, 2, 3,
    4, 5, 1, ..., 4,  0, 0, ..., 0]].

    Args:
        mask: mask value with the same type as an element in indexed sentences
        sentences (iter): iterable of original sentences
        indexed_sentences (iter): iterable of indexed (numeric) sentences
        rand_state (int): random seed

    Returns:
        iter: iterable of masked indexed (numeric) sentences
    """

    random.seed(rand_state)
    masked_indexed_sentences = copy.deepcopy(indexed_sentences)
    masked_values = []

    for s, i in zip(sentences, masked_indexed_sentences):

        # Note that the SOS, EOS and padding shall not be masked
        index = random.randint(1, len(s))

        masked_values.append(i[index])
        i[index] = type(i[0])(mask)

    return masked_values, masked_indexed_sentences
