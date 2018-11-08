""" 
    File Name:          DLTM/masked_smiles_pred.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/8/18
    Python Version:     3.6.6
    File Description:   

"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks.modules.encoder import Encoder
from util.datasets.smiles_dataset import SMILESDataset
from util.misc.rand_state_seeding import seed_random_state


def train(encoder, clf, device, train_loader, optimizer, epoch):

    encoder.train()
    clf.train()

    for batch_idx, (mask, masked, value) in enumerate(train_loader):
        mask, masked, value = \
            mask.to(device), masked.to(device), value.to(device)
        optimizer.zero_grad()
        temp = encoder(masked, mask.unsqueeze(-2))
        output = clf(temp.view(value.size(0), -1))
        output = F.log_softmax(output, dim=-1)

        loss = F.nll_loss(output, value)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(masked), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(encoder, clf, device, test_loader):

    encoder.eval()
    clf.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for mask, masked, value in test_loader:
            mask, masked, value = \
                mask.to(device), masked.to(device), value.to(device)

            temp = encoder(masked, mask.unsqueeze(-2))
            output = clf(temp.view(value.size(0), -1))
            output = F.log_softmax(output, dim=-1)
            test_loss += F.nll_loss(output,
                                    value,
                                    reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(value.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Masked SMILES string prediction with transformer encoder')

    # Encoder parameters ######################################################
    parser.add_argument('--seq_length', type=int, default=128,
                        help='max length for training/testing SMILES strings')
    parser.add_argument('--pos_freq', type=float, default=100.0,
                        help='frequency for positional encoding')
    parser.add_argument('--embedding_scale', type=float, default=16.0,
                        help='scale of word embedding to positional encoding')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='embedding and model dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of encoding layers in encoder')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads in multi-head attention layer')
    parser.add_argument('--ff_mid_dim', type=int, default=128,
                        help='dimension of mid layer in feed forward module')

    parser.add_argument('--pe_dropout', type=float, default=0.1,
                        help='dropout of positional encoding module')
    parser.add_argument('--mha_dropout', type=float, default=0.1,
                        help='dropout of multi-head attention module')
    parser.add_argument('--ff_dropout', type=float, default=0.1,
                        help='dropout of feed forward module')
    parser.add_argument('--enc_dropout', type=float, default=0.1,
                        help='dropout between encoder layers')

    # Training/validation parameters ##########################################
    parser.add_argument('--trn_batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation')
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                        help='ratio of validation dataset over all data')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer for transformer encoder training',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate of the optimizer')

    # Miscellaneous config ####################################################
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()
    print('Training Arguments:\n' + json.dumps(vars(args), indent=4))

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data loaders for training/validation
    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        'num_workers': 4 if use_cuda else 0,
        'pin_memory': True if use_cuda else False, }

    trn_loader = torch.utils.data.DataLoader(
        SMILESDataset(data_root='./data/',
                      training=True,
                      rand_state=args.rand_state,
                      seq_len=args.seq_length,
                      val_ratio=args.validation_ratio),
        batch_size=args.trn_batch_size, **dataloader_kwargs)

    val_loader = torch.utils.data.DataLoader(
        SMILESDataset(data_root='./data/',
                      training=False,
                      rand_state=args.rand_state,
                      seq_len=args.seq_length,
                      val_ratio=args.validation_ratio),
        batch_size=args.trn_batch_size, **dataloader_kwargs)

    # Model and optimizer
    dict_size = len(trn_loader.dataset.encode_dict)
    encoder = Encoder(dict_size=dict_size,
                      seq_length=args.seq_length,

                      base_feq=args.pos_freq,
                      emb_scale=args.embedding_scale,

                      emb_dim=args.embedding_dim,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      ff_mid_dim=args.ff_mid_dim,

                      pe_dropout=args.pe_dropout,
                      mha_dropout=args.mha_dropout,
                      ff_dropout=args.ff_dropout,
                      enc_dropout=args.enc_dropout).to(device)

    clf = nn.Sequential(nn.Linear(args.embedding_dim * args.seq_length,
                                  dict_size)).to(device)

    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    parameters = list(encoder.parameters()) + list(clf.parameters())

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(parameters, lr=args.lr)
    else:
        optimizer = optim.Adam(parameters, lr=args.lr)

    for epoch in range(1, args.max_num_epochs + 1):
        train(encoder, clf, device, trn_loader, optimizer, epoch)
        test(encoder, clf, device, val_loader)


if __name__ == '__main__':
    main()
