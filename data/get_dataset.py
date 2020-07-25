#Code for downloading and exporting the training and test datasets used in PSP

import os
import numpy as np
import shutil
import gzip
import requests

#File paths for train and test datasets
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'cb513+profile_split1.npy.gz'
TRAIN_NPY = 'cullpdb+profile_6133_filtered.npy'
TEST_NPY = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'

#URL's for train and test dataset download
TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"
CASP_10_URL = "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5"
CASP_11_URL = "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5"

#turn into class and add asertions
#download and unzip filtered cullpdb training data
def get_cullpdb_filtered():

    print('Downloading Cullpdb 6133 dataset...\n')

    try:
        if not (os.path.isfile(TRAIN_PATH)):
            r = requests.get(TRAIN_URL, allow_redirects = True)
            open(TRAIN_PATH, 'wb').write(r.content)
            dir_path = os.path.dirname(os.path.realpath(TRAIN_PATH))
            source_path = dir_path + '/' + TRAIN_PATH
            destination_path = dir_path + '/' + TRAIN_NPY

            print('Exporting Cullpdb 6133 datatset....')
            with gzip.open(TRAIN_PATH, 'rb') as f_in:
                with open(TRAIN_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('Cullpdb 6133 dataset already present...\n')

    except OSError:
        print('Error downloading and exporting training dataset\n')
        return

#download and unzip CB513 test data
def get_cb513():

    print('Downloading CB513 dataset...\n')

    try:
        if not (os.path.isfile(TEST_PATH)):
            #os.system(f'wget -O {TEST_PATH} {TEST_URL}')
            r = requests.get(TEST_URL, allow_redirects = True)
            open(TEST_PATH, 'wb').write(r.content)
            dir_path = os.path.dirname(os.path.realpath(TEST_PATH))
            source_path = dir_path + '/' + TEST_PATH
            destination_path = dir_path + '/' + TEST_NPY

            print('Exporting CB513 datatset...\n')
            with gzip.open(TEST_PATH, 'rb') as f_in:
                with open(TEST_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('CB513 dataset already present...\n')

    except OSError:
        print('Error downloading and exporting testing dataset\n')
        return

#downloading CASP10 test dataset
def get_casp10():

    try:
        if not (os.path.isfile(CASP10_PATH)):
            #os.system(f'wget -O {CASP10_PATH} {CASP_10_URL}')
            #os.system('wget -O {} {}'.format(CASP10_PATH, CASP_10_URL))
            r = requests.get(CASP_10_URL, allow_redirects = True)
            open('casp10.h5', 'wb').write(r.content)
            print('CASP10 dataset downloaded\n')

        else:
            print('CASP10 dataset already exists\n')

    except OSError:
        print('Error downloading and exporting CASP10 dataset\n')

#downloading CASP11 test dataset
def get_casp11():

    try:
        if not (os.path.isfile(CASP11_PATH)):
            #os.system(f'wget -O {CASP10_PATH} {CASP_10_URL}') #errors using wget so using requests
            #os.system('wget -O {} {}'.format(CASP10_PATH, CASP_10_URL))
            r = requests.get(CASP_11_URL, allow_redirects = True)
            open('casp11.h5', 'wb').write(r.content)
            print('CASP11 dataset downloaded\n')

        else:
            print('CASP11 dataset already exists\n')

    except OSError:
        print('Error downloading and exporting CASP11 dataset\n')
