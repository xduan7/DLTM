""" 
    File Name:          DLTM/pcba.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/15/18
    Python Version:     3.6.6
    File Description:   

"""
from deepchem.molnet import load_pcba

PCBA_tasks, (train, valid, test), transformers = \
    load_pcba(featurizer='Raw',
              split='scaffold',
              reload=True,)

print(PCBA_tasks)
print(train)
