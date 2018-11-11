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
    """create_path('../../some_path')

    This function tries to create a directory if it does not exist.

    Args:
        path (str): string for path/dir to be created.

    Returns:
        None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
