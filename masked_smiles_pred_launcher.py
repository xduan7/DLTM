""" 
    File Name:          DLTM/masked_smiles_pred_launcher.py
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

        # # Baseline ############################################################
        # {'pos_freq': '4.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '128',
        #  'num_layers': '6',
        #  'num_heads': '8', },
        #
        # # Test out freq vs performance ########################################
        # {'pos_freq': '2.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '128',
        #  'num_layers': '6',
        #  'num_heads': '8', },
        #
        # {'pos_freq': '8.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '128',
        #  'num_layers': '6',
        #  'num_heads': '8', },
        #
        # # Test out dimension vs performance ###################################
        # {'pos_freq': '4.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '64',
        #  'num_layers': '6',
        #  'num_heads': '8', },
        #
        # {'pos_freq': '4.0',
        #  'embedding_scale': '16.0',
        #  'embedding_dim': '256',
        #  'num_layers': '6',
        #  'num_heads': '8', },

        # Number of layers vs performance #####################################
        {'pos_freq': '4.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '12',
         'num_heads': '8', },

        {'pos_freq': '4.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '3',
         'num_heads': '8', },

        # Number of heads vs performance ######################################
        {'pos_freq': '4.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '6',
         'num_heads': '4', },

        {'pos_freq': '4.0',
         'embedding_scale': '16.0',
         'embedding_dim': '128',
         'num_layers': '6',
         'num_heads': '16', },

    ]

    for param_dict in param_dict_list:

        now = datetime.datetime.now()

        # Save log with timestamp name
        tee = Tee('./results/logs/masked_smiles_pred/%02d%02d_%02d%02d.txt'
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
            '--max_num_epochs', '500',

            '--optimizer', 'Adam',
            '--lr', '0.001',
            '--l2_regularization', '1e-5',
            '--lr_decay_factor', '1.0',
            '--num_logs_per_epoch', '6',
            '--early_stop_patience', '10',

            # Miscellaneous config ############################################
            '--rand_state', '0', ]

        runpy.run_module('masked_smiles_pred')
        sys.stdout = tee.default_stdout()
