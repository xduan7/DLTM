""" 
    File Name:          DLTM/path_creating.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/7/18
    Python Version:     3.6.6
    File Description:   

"""

import os


def create_path(path: str):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
