""" 
    File Name:          DLTM/protein_func_pred_launcher.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/17/18
    Python Version:     3.6.6
    File Description:   

"""

import sys
import runpy
import datetime

from util.misc.tee import Tee


if __name__ == '__main__':

    param_dict_list = [

        # Baseline ############################################################
        {
            'seq_length': '512',
            'embedding_dim': '8',
            'num_layers': '6',
            'num_heads': '4',
            'dropout': '0.0',
         },

    ]

    for param_dict in param_dict_list:

        now = datetime.datetime.now()

        # Save log with timestamp name
        tee = Tee('./results/logs/protein_func_pred/%02d%02d_%02d%02d.txt'
                  % (now.month, now.day, now.hour, now.minute))
        sys.stdout = tee

        sys.argv = [
            'protein_func_pred',

            # Encoder parameters ##############################################
            '--seq_length', param_dict['seq_length'],
            '--pos_freq', '32.0',
            '--embedding_scale', '16.0',

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
            '--max_num_epochs', '500',

            '--optimizer', 'Adam',
            '--lr', '0.0004',
            '--l2_regularization', '1e-5',
            '--lr_decay_factor', '0.95',
            '--num_logs_per_epoch', '6',
            '--early_stop_patience', '10',

            # Miscellaneous config ############################################
            '--rand_state', '0', ]

        runpy.run_module('protein_func_pred')
        sys.stdout = tee.default_stdout()
