""" 
    File Name:          DLTM/pcba.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/15/18
    Python Version:     3.6.6
    File Description:   

"""
import deepchem
from deepchem.data import NumpyDataset
from rdkit import Chem
from deepchem.molnet import load_pcba, load_hiv

tasks, (train, valid, test), transformers = \
    load_hiv(featurizer='Raw',
             split='scaffold',
             reload=True,)

tmp = train.X
print(tmp)


iter_batches = train.iterbatches(batch_size=1024 * 8, deterministic=False)
# molecules, targets, smiles = next(iter_batches)
X, y, w, ids = next(iter_batches)

y_cnt = 0
w_cnt = 0
print(ids[0])
print(X[0].GetAtoms())

print(Chem.MolToMolBlock(X[0]))


atoms = X[0].GetAtoms()

for a in atoms:
    print(a.GetSymbol())

# for i in range(1024 * 8):
#
#     print('ID: %s', ids[i])
#     print('y: %s', y[i])
#     print('w: %s', w[i])
#
#     if y[i] != 0.:
#         y_cnt += 1
#
#     if w[i] != 1.:
#         w_cnt += 1
#
# print(y_cnt)
# print(w_cnt)

# print(len(ids[0]))
# print(w[0])
# print(len(w[0]))



#
# print(molecules[0])
# print(molecules[0].GetAtoms())


# print(PCBA_tasks)
# print(len(train))
#
# tmp = NumpyDataset(X=train)
#
# train.iterbatches(batch_size=32, deterministic=True)
#
# tmp.from_DiskDataset(train)
#
# print(tmp)
