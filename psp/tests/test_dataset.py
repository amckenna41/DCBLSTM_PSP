
##########################
### Tests for datasets ###
##########################

from data.get_dataset import *
from data.load_dataset import *
import os
import requests
import numpy as np

#File paths for train and test datasets
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'cb513+profile_split1.npy.gz'
TRAIN_NPY = 'cullpdb+profile_6133_filtered.npy'
TEST_NPY = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'


#Test URL status codes
def test_dataset_url():

    #test URL endpoint for dataset is active and not errorenous
    r = requests.get(TRAIN_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(TEST_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(CASP_10_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(CASP_11_URL, allow_redirects = True)
    assert(r.status_code == 200)


def test_create_dataset_instances():
    pass

def test_download_existing_datasets():
    pass

def test_cullpdb():
    pass

def test_cb513():
    pass

def test_casp10():
    pass

def test_casp11():
    pass

#
# def test_get_datasets():
#
#     get_cullpdb_filtered()
#     get_cb513()
#     get_casp10()
#     get_casp11()
#
#     #testing file exists after getting and downloading it locally
#     assert(os.path.isfile(os.getcwd() + '/' + TRAIN_NPY))
#     assert(os.path.isfile(os.getcwd() + '/' + TEST_NPY))
#     assert(os.path.isfile(os.getcwd() + '/' + CASP10_PATH))
#     assert(os.path.isfile(os.getcwd() + '/' + CASP11_PATH))
#
# #tests for the training dataset
# def test_load_cul6133_filtered():
#
#     train_hot,train_pssm,train_label, val_hot,val_pssm,val_label = load_cul6133_filted()
#
#     #test correct shape and size of datasets
#     assert(train_hot.shape == (5278, 700))
#     assert(type(train_hot).__module__ == (np.__name__))
#
#     assert(train_pssm.shape == (5278, 700, 21))
#     assert(type(train_pssm).__module__ == (np.__name__))
#
#     assert(train_label.shape == (5278, 700, 8))
#     assert(type(train_label).__module__ == (np.__name__))
#
#     assert(val_hot.shape == (256, 700))
#     assert(type(val_hot).__module__ == (np.__name__))
#
#     assert(val_pssm.shape == (256, 700, 21))
#     assert(type(val_pssm).__module__ == (np.__name__))
#
#     assert(val_label.shape == (256, 700, 8))
#     assert(type(val_label).__module__ == (np.__name__))
#
#     #load half of the training dataset and check correct shapes of data
#     all_data = 0.5
#     train_hot,train_pssm,train_label, val_hot,val_pssm,val_label = load_cul6133_filted(all_data=all_data)
#
#
#     assert(train_hot.shape == (2639, 700))
#
#     assert(train_pssm.shape == (2639, 700, 21))
#
#     assert(train_label.shape == (2639, 700, 8))
#
#     assert(val_hot.shape == (128, 700))
#
#     assert(val_pssm.shape == (128, 700, 21))
#
#     assert(val_label.shape == (128, 700, 8))
#
#     #load quarter of the training dataset and check correct shapes of data
#     all_data = 0.25
#     train_hot,train_pssm,train_label, val_hot,val_pssm,val_label = load_cul6133_filted(all_data=all_data)
#
#     assert(train_hot.shape == (1319, 700))
#
#     assert(train_pssm.shape == (1319, 700, 21))
#
#     assert(train_label.shape == (1319, 700, 8))
#
#     assert(val_hot.shape == (64, 700))
#
#     assert(val_pssm.shape == (64, 700, 21))
#
#     assert(val_label.shape == (64, 700, 8))
#
# #tests for the CB513 test dataset
# def test_load_cb513():
#
#     test_hot, testpssm, testlabel = load_cb513()
#
#     #test shapes and data types of test dataset
#     assert(test_hot.shape == (514, 700))
#     assert(type(test_hot).__module__ == (np.__name__))
#
#     assert(testpssm.shape == (514, 700, 21))
#     assert(type(testpssm).__module__ == (np.__name__))
#
#     assert(testlabel.shape == (514, 700, 8))
#     assert(type(testlabel).__module__ == (np.__name__))
#
# #tests for the CASP10 test dataset
# def test_load_casp10():
#
#     casp10_data_test_hot, casp10_data_pssm, test_labels = load_casp10()
#
#     #test shapes and data types of test dataset
#     assert(casp10_data_test_hot.shape == (123, 700))
#     assert(type(casp10_data_test_hot).__module__ == (np.__name__))
#
#     assert(casp10_data_pssm.shape == (123, 700, 21))
#     assert(type(casp10_data_pssm).__module__ == (np.__name__))
#
#     assert(test_labels.shape == (123, 700, 8))
#     assert(type(test_labels).__module__ == (np.__name__))
#
# #tests for the CASP11 test dataset
# def test_load_casp11():
#
#     casp11_data_test_hot, casp11_data_test_pssm, test_labels = load_casp11()
#
#     #test shapes and data types of test dataset
#     assert(casp11_data_test_hot.shape == (105, 700))
#     assert(type(casp11_data_test_hot).__module__ == (np.__name__))
#
#     assert(casp11_data_test_pssm.shape == (105, 700, 21))
#     assert(type(casp11_data_test_pssm).__module__ == (np.__name__))
#
#     assert(test_labels.shape == (105, 700, 8))
#     assert(type(test_labels).__module__ == (np.__name__))
#
#
# if __name__ == "__main__":
#
#     test_dataset_url()
#     print("URL tests passed")
#
#     test_get_datasets()
#     print("Get dataset tests passed")
#
#     test_load_cb513()
#     print("Load CB513 test passed")
#
#     test_load_casp10()
#     print("Load CASP10 test passed")
#
#     test_load_casp11()
#     print("Load CASP11 test passed\n")
#     print("Everything passed")
