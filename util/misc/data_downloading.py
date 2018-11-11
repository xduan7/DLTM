""" 
    File Name:          DLTM/data_downloading.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               11/5/18
    Python Version:     3.6.6
    File Description:   

"""

import os
import errno
import logging
import urllib.request


FTP_ROOT = 'http://bioseed.mcs.anl.gov/~fangfang/DREAM/'
logger = logging.getLogger(__name__)


def download_files(file_names: str or iter,
                   target_dir: str,
                   ftp_root: str = FTP_ROOT):
    """download_files(['some', 'file', 'names'], './data/, 'ftp://some-server')

    This function download one or more files from given FTP server to target
    folder. Note that the file names wil be the same with FTP server.

    Args:
        file_names (str or iter): a string of filename or an iterable structure
            of multiple file names for downloading.
        target_dir (str): target folder for storing downloaded data.
        ftp_root (str): address for FTP server.

    Returns:
        None
    """

    if type(file_names) is str:
        file_names = [file_names, ]

    # Create  target folder if not exist
    try:
        os.makedirs(target_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger.error('Failed to create data folders', exc_info=True)
            raise

    # Download each file in the list
    for file_name in file_names:
        file_path = os.path.join(target_dir, file_name)

        if not os.path.exists(file_path):
            logger.debug('File does not exit. Downloading %s ...' % file_name)

            url = ftp_root + file_name
            try:
                url_data = urllib.request.urlopen(url)
                with open(file_path, 'wb') as f:
                    f.write(url_data.read())
            except IOError:
                logger.error('Failed to open and download url %s.' % url,
                             exc_info=True)
                raise


# Test segment
if __name__ == '__main__':
    download_files('dtc.train.filtered.txt', '../../data/raw/')

