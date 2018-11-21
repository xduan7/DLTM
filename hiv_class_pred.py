""" 
    File Name:          DLTM/hiv_class_pred.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/21/18
    Python Version:     3.6.6
    File Description:   

"""

import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR

from networks.encoder_clf import EncoderClf
from networks.modules.encoder import Encoder
from util.datasets.hiv_dataset import HIVDataset
from util.misc.optimizer import get_optimizer
from util.misc.rand_state_seeding import seed_random_state


def train(clf, device, trn_loader, optimizer, num_logs_per_epoch):

    clf.train()
    num_batches_per_log = np.floor(len(trn_loader) / num_logs_per_epoch)

    recorded_loss = 0.

    for batch_index, (mask, data, target) in enumerate(trn_loader):

        mask, data, target = \
            mask.to(device), data.to(device), target.to(device)

        optimizer.zero_grad()
        output = torch.sigmoid(clf(data, mask))
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        recorded_loss += loss.item()

        if (batch_index + 1) % num_batches_per_log == 0:
            print('\t Progress: %.1f%% \t Loss: %.4f'
                  % (100. * (batch_index + 1) / len(trn_loader),
                     recorded_loss / num_batches_per_log))
            recorded_loss = 0.


def validate(clf, device, val_loader, printing=True):

    clf.eval()

    val_loss = 0.
    val_correct = 0

    y_true = np.array([])
    y_score = np.array([])

    with torch.no_grad():

        for mask, data, target in val_loader:

            mask, data, target = \
                mask.to(device), data.to(device), target.to(device)

            output = torch.sigmoid(clf(data, mask))
            loss = F.binary_cross_entropy(
                output, target, reduction='sum').item()

            val_loss += loss
            prediction = output.max(1, keepdim=True)[1]

            target = target.type(torch.LongTensor).view_as(prediction)
            val_correct += prediction.eq(target.to(device)).sum().item()

            y_true = np.concatenate(
                (y_true, target.cpu().numpy().reshape(-1)), axis=0)
            y_score = np.concatenate(
                (y_score, output.cpu().numpy().reshape(-1)), axis=0)

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    roc_score = roc_auc_score(y_true=y_true, y_score=y_score)

    if printing:
        print('\nValidation Results: \n'
              '\t Average Loss: %.4f, '
              '\t Accuracy: %6i/%6i (%.1f%%), '
              '\t ROC-AUC Score: %.4f\n'
              % (val_loss, val_correct, len(val_loader.dataset),
                 100. * val_acc, roc_score))

    return val_acc, roc_score


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='HIV drug class prediction with transformer encoder')

    # Encoder parameters ######################################################
    parser.add_argument('--seq_length', type=int, default=256,
                        help='max length for training/testing SMILES strings')
    parser.add_argument('--pos_freq', type=float, default=4.0,
                        help='frequency for positional encoding')
    parser.add_argument('--embedding_scale', type=float, default=16.0,
                        help='scale of word embedding to positional encoding')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='embedding and model dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of encoding layers in encoder')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads in multi-head attention layer')
    parser.add_argument('--ff_mid_dim', type=int, default=32,
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
    parser.add_argument('--tst_batch_size', type=int, default=256,
                        help='input batch size for testing')

    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer for transformer encoder training',
                        choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate of the optimizer')
    parser.add_argument('--l2_regularization', type=float, default=1e-4,
                        help='L2 regularization for nn weights')
    parser.add_argument('--lr_decay_factor', type=float, default=1.0,
                        help='decay factor for learning rate')
    parser.add_argument('--num_logs_per_epoch', type=int, default=5,
                        help='number of logs per epoch during training')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='number of epochs for early stopping if no '
                             'improvement')

    # Miscellaneous config ####################################################
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()
    print('Training Arguments:\n' + json.dumps(vars(args), indent=4))
    print('\n' + '=' * 80 + '\n')

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (cuda or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data loaders for training/validation
    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        'num_workers': 8 if use_cuda else 0,
        'pin_memory': True if use_cuda else False, }

    trn_loader = torch.utils.data.DataLoader(
        HIVDataset(
            token_dict_path='./data/HIV_atom_token_dict.json',
            tokenized_data_path='./data/HIV_trn_tokenized_on_atom.pkl',
            dataset_usage='training',
            max_seq_length=args.seq_length),
        batch_size=args.trn_batch_size, **dataloader_kwargs)

    val_loader = torch.utils.data.DataLoader(
        HIVDataset(
            token_dict_path='./data/HIV_atom_token_dict.json',
            tokenized_data_path='./data/HIV_val_tokenized_on_atom.pkl',
            dataset_usage='validation',
            max_seq_length=args.seq_length),
        batch_size=args.trn_batch_size, **dataloader_kwargs)

    tst_loader = torch.utils.data.DataLoader(
        HIVDataset(
            token_dict_path='./data/HIV_atom_token_dict.json',
            tokenized_data_path='./data/HIV_tst_tokenized_on_atom.pkl',
            dataset_usage='test',
            max_seq_length=args.seq_length),
        batch_size=args.trn_batch_size, **dataloader_kwargs)

    # Model and optimizer
    dict_size = len(trn_loader.dataset.token_dict)

    # Using
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
    output_layer = nn.Sequential(
        nn.Linear(args.embedding_dim * args.seq_length, 1)).to(device)

    clf = EncoderClf(encoder=encoder, output_module=output_layer)

    optimizer = get_optimizer(opt_type=args.optimizer,
                              networks=clf,
                              learning_rate=args.lr,
                              l2_regularization=args.l2_regularization)

    lr_decay = LambdaLR(optimizer=optimizer,
                        lr_lambda=lambda e:
                        args.lr_decay_factor ** e)

    best_roc_score = 0.0
    test_roc_score = 0.0
    patience = 0

    for epoch in range(1, args.max_num_epochs + 1):

        print('Epoch %3i: ' % epoch)

        train(clf, device, trn_loader, optimizer, args.num_logs_per_epoch)
        val_acc, roc_score = validate(clf, device, val_loader)

        if roc_score > best_roc_score:

            best_roc_score = roc_score
            patience = 0

            test_acc, test_roc_score = \
                validate(clf, device, tst_loader, printing=False)
        else:
            patience += 1

        if patience >= args.early_stop_patience:
            break

        lr_decay.step(epoch)
        print('=' * 80 + '\n')

    print('Best ROC-AUC: %.4f for validation and %.4f for test'
          % (best_roc_score, test_roc_score))


main()
