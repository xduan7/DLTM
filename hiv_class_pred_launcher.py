""" 
    File Name:          DLTM/hiv_class_pred_launcher.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/21/18
    Python Version:     3.6.6
    File Description:   

"""

import sys
import runpy
import datetime

from util.misc.tee import Tee


if __name__ == '__main__':

    param_dict_list = [

        # # Baseline for prediction
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},

        #######################################################################
        # # Positional encoding frequency #######################################
        # {'pos_freq': '8.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '32.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Embedding scale #####################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Embedding dimension #################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '16',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '64',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.1'},

        # #######################################################################
        # Number of layers ####################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '3',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Number of heads #####################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '4',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '16',
        #  'dropout': '0.1'},
        #
        # # Dropout #############################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.05'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '32',
        #  'num_layers': '6',
        #  'num_heads': '8',
        #  'dropout': '0.2'},

        # # More ##############################################################
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Freq
        # {'pos_freq': '12.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '20.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Dim
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '48',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '72',
        #  'num_layers': '9',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Layers
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '8',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '10',
        #  'num_heads': '8',
        #  'dropout': '0.1'},
        #
        # # Heads
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '9',
        #  'num_heads': '4',
        #  'dropout': '0.1'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '4.0',
        #  'embedding_dim': '64',
        #  'num_layers': '9',
        #  'num_heads': '16',
        #  'dropout': '0.1'},

        # # 512 sequence
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '48',
        #  'num_layers': '10',
        #  'num_heads': '6',
        #  'dropout': '0.1',
        #  'l2_regularization': '1e-5'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '48',
        #  'num_layers': '10',
        #  'num_heads': '6',
        #  'dropout': '0.2',
        #  'l2_regularization': '1e-5'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '48',
        #  'num_layers': '10',
        #  'num_heads': '6',
        #  'dropout': '0.1',
        #  'l2_regularization': '1e-4'},
        #
        # {'pos_freq': '16.0',
        #  'embedding_scale': '8.0',
        #  'embedding_dim': '48',
        #  'num_layers': '10',
        #  'num_heads': '6',
        #  'dropout': '0.2',
        #  'l2_regularization': '1e-4'},

        # {'pos_freq': '16.0',
        #  'embedding_scale': '20.0',
        #  'embedding_dim': '64',
        #  'num_layers': '8',
        #  'num_heads': '4',
        #  'dropout': '0.0',
        #  'l2_regularization': '0'},

        {'pos_freq': '16.0',
         'embedding_scale': '20.0',
         'embedding_dim': '32',
         'num_layers': '6',
         'num_heads': '4',
         'dropout': '0.1',
         'l2_regularization': '1e-5'},

        {'pos_freq': '16.0',
         'embedding_scale': '20.0',
         'embedding_dim': '32',
         'num_layers': '6',
         'num_heads': '4',
         'dropout': '0.2',
         'l2_regularization': '1e-4'},

        {'pos_freq': '16.0',
         'embedding_scale': '20.0',
         'embedding_dim': '32',
         'num_layers': '4',
         'num_heads': '4',
         'dropout': '0.1',
         'l2_regularization': '1e-5'},

    ]

    for param_dict in param_dict_list:

        now = datetime.datetime.now()

        # Save log with timestamp name
        tee = Tee('./results/logs/hiv_class_pred/%02d%02d_%02d%02d.txt'
                  % (now.month, now.day, now.hour, now.minute))
        sys.stdout = tee

        sys.argv = [
            'hiv_class_pred',

            # Encoder parameters ##############################################
            '--seq_length', '700',
            '--pos_freq', param_dict['pos_freq'],
            '--embedding_scale', param_dict['embedding_scale'],

            '--embedding_dim', param_dict['embedding_dim'],
            '--num_layers', param_dict['num_layers'],
            '--num_heads', param_dict['num_heads'],
            '--ff_mid_dim', param_dict['embedding_dim'],

            '--pe_dropout', param_dict['dropout'],
            '--mha_dropout', param_dict['dropout'],
            '--ff_dropout', param_dict['dropout'],
            '--enc_dropout', param_dict['dropout'],

            # Training/validation parameters ##################################
            '--trn_batch_size', '32',
            '--val_batch_size', '256',
            '--tst_batch_size', '256',
            '--max_num_epochs', '500',

            '--optimizer', 'Adam',
            '--lr', '0.00004',
            '--l2_regularization', param_dict['l2_regularization'],
            '--lr_decay_factor', '0.98',
            '--num_logs_per_epoch', '5',
            '--early_stop_patience', '20',

            # Miscellaneous config ############################################
            '--multi_gpu',
            '--rand_state', '0', ]

        runpy.run_module('hiv_class_pred')
        sys.stdout = tee.default_stdout()
