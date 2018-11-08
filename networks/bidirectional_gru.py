""" 
    File Name:          DLTM/bidirectional_gru.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/6/18
    Python Version:     3.6.6
    File Description:   

"""


import numpy as np
import torch, torch.nn as nn
from torch import optim

from util.datasets.smiles_dataset import SMILESDataset




smiles_trn_loader = torch.utils.data.DataLoader(
    SMILESDataset('../data/raw/', True, rand_state=1),
        batch_size=32)



bi_grus = torch.nn.RNN(input_size=46,
                       hidden_size=128,
                       num_layers=2,
                       batch_first=True,
                       bidirectional=True).cuda()

optimizer = optim.SGD(bi_grus.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    for i, data in enumerate(smiles_trn_loader):
        # get the inputs
        mask_index, smiles, encoded_smiles, \
        masked_smiles, masked_encoded_smiles = data

        masked_encoded_smiles = masked_encoded_smiles.float().cuda()
        print(masked_encoded_smiles.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        h0 = torch.randn(4, 32, 128).cuda()
        output, hn = bi_grus(masked_encoded_smiles, h0)

        print(output.shape)
        print(hn.shape)

        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

