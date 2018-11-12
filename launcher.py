""" 
    File Name:          DLTM/launcher.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/12/18
    Python Version:     3.6.6
    File Description:   

"""

import sys
import runpy
import datetime

from util.misc.tee import Tee


if __name__ == '__main__':

    param_dict_list = [

        # Test out freq vs performance
        {'pos_freq': '4.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '6',
         'num_heads': '8', },

        {'pos_freq': '8.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '6',
         'num_heads': '8', },

        {'pos_freq': '16.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '6',
         'num_heads': '8', }, ]

    for param_dict in param_dict_list:

        now = datetime.datetime.now()

        # Save log with timestamp name
        tee = Tee('./results/logs/%02d%02d_%02d%02d.txt'
                  % (now.month, now.day, now.hour, now.minute))
        sys.stdout = tee

        sys.argv = [
            'masked_smiles_pred',

            # Encoder parameters ##############################################
            '--seq_length', '128',
            '--pos_freq', param_dict['pos_freq'],
            '--embedding_scale', param_dict['embedding_scale'],

            '--embedding_dim', param_dict['embedding_dim'],
            '--num_layers', param_dict['num_layers'],
            '--num_heads', param_dict['num_heads'],
            '--ff_mid_dim', param_dict['embedding_dim'],

            '--pe_dropout', '0.1',
            '--mha_dropout', '0.1',
            '--ff_dropout', '0.1',
            '--enc_dropout', '0.1',

            # Training/validation parameters ##################################
            '--trn_batch_size', '32',
            '--val_batch_size', '256',
            '--validation_ratio', '0.1',
            '--max_num_epochs', '100',

            '--optimizer', 'SGD',
            '--lr', '0.001',
            '--l2_regularization', '1e-5',
            '--lr_decay_factor', '0.95',
            '--num_logs_per_epoch', '5',

            # Miscellaneous config ############################################
            '--rand_state', '0', ]

        runpy.run_module('masked_smiles_pred')
        sys.stdout = tee.default_stdout()
