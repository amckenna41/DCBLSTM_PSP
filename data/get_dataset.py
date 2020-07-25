#Code for downloading and exporting the training and test datasets used in PSP

import os
import numpy as np
import subprocess
from utils import load_gz
import shutil

#File paths for train and test datasets
TRAIN_PATH = '/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '/cb513+profile_split1.npy.gz'
TRAIN_NPY = '/cullpdb+profile_6133_filtered.npy'
TEST_NPY = '/cb513+profile_split1.npy'

#URL's for train and test dataset download
TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"
#CASP_10 URL =
#CASP_11 URL =

#download datasets
#turn into class and add asertions
def download_export_dataset():
    print('Downloading Cullpdb 6133 dataset ...')

    try:
        if not (os.path.isfile(TRAIN_PATH)):
        #os.makedirs('../dataset', exist_ok=True) - create dataset directory if not created
            os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')

            dir_path = os.path.dirname(os.path.realpath(TRAIN_PATH))
            source_path = dir_path + '/' + TRAIN_PATH
            destination_path = dir_path + '/' + TRAIN_NPY

            with gzip.open(TRAIN_PATH, 'rb') as f_in:
                with open(TRAIN_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('Cullpdb 6133 dataset already present...')

    except OSError:
        print('Error downloading and exporting training dataset')
        return

    print('Downloading CB513 dataset ...')

    try:
        if not (os.path.isfile(TEST_PATH)):
            os.system(f'wget -O {TEST_PATH} {TEST_URL}')

            dir_path = os.path.dirname(os.path.realpath(TEST_PATH))
            source_path = dir_path + '/' + TEST_PATH
            destination_path = dir_path + '/' + TEST_NPY

            with gzip.open(TEST_PATH, 'rb') as f_in:
                with open(TEST_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('CB513 dataset already present...')

    except OSError:
            print('CB513 dataset already in directory')


    #casp10, casp11 data
