#Code for getting and downloading the training and test datasets used in PSP

import os
import numpy as np
import subprocess
from utils import load_gz, save_text, save_picke
import shutil

#File paths for train and test datasets
TRAIN_PATH = '../dataset/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../dataset/cb513+profile_split1.npy.gz'

#URL's for train and test dataset download
TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"

#download datasets
def download_dataset():
    print('Downloading Cullpdb 6133 dataset ...')

    try:
        if not (os.path.isfile(TRAIN_PATH)):
        os.makedirs('../dataset', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
    except:

    print('Downloading CB513 dataset ...')
    try:
        if not (os.path.isfile(TEST_PATH)):
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')
    except:


        #casp10, casp11 data
    export_dataset()

#Export the zipped dataset files
def export_dataset():

    #Export Cullpdb 6133 training data
    filename = '../dataset/cullpdb+profile_6133_filtered.npy'
    #get path of dataset in cwd
    dir_path = os.path.dirname(os.path.realpath(filename))
    source_path = dir_path + '/' + filename
    destination_path = dir_path + '/' + new_filename

    #open zipped file and copy contents to new .npy file
    with gzip.open(filename, 'rb') as f_in:
        with open(new_filename, 'wb') as f_out:
            shutil.copyfile(source_path, destination_path)


    filename = '../dataset/cb513+profile_split1.npy'

    #get path of dataset in cwd
    dir_path = os.path.dirname(os.path.realpath(filename))
    source_path = dir_path + '/' + filename
    destination_path = dir_path + '/' + new_filename

    #open zipped file and copy contents to new .npy file
    with gzip.open(filename, 'rb') as f_in:
        with open(new_filename, 'wb') as f_out:
            shutil.copyfile(source_path, destination_path)
