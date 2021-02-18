
#########################################################################
###                       Tests for datasets                          ###
#########################################################################

try:
    from data.load_dataset import *
except:
    from psp.data.load_dataset import *
import os
import requests
import numpy as np
import globals

def test_dataset_url():

    """
    Description:
        Testing URL endpoints for datasets - testing if successful http request status code 200 returned.
    Args:
        None
    Returns:
        Test passed or test failed

    """
    r = requests.get(TRAIN_FILTERED_6133_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(TRAIN_UNFILTERED_6133_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(TRAIN_FILTERED_5926_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(TRAIN_UNFILTERED_5926_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(CB513_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(CASP10_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)

    r = requests.get(CASP11_URL, allow_redirects = True)
    assert r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code)


def test_cullpdb():

    """
    Description:
        Testing Cullpdb filtered/unfiltered training dataset class and all its associated functions
    Args:
        None
    Returns:
        Test passed or test failed

    """

    ### CullPDB 6133 filtered dataset tests ###
    cullpdb =  CullPDB(6133, filtered = True)

    assert cullpdb.train_hot.shape == (5278, 700), 'Data shape must be (5278x700)'
    assert cullpdb.trainpssm.shape == (5278, 700, 21),'Data shape must be (5278x700x21)'
    assert cullpdb.trainlabel.shape == (5278, 700, 8),'Data shape must be (5278x700x21)'

    assert type(cullpdb.train_hot).__module__ == (np.__name__),'Data type must be numpy array'
    assert type(cullpdb.trainpssm).__module__ == (np.__name__),'Data type must be numpy array'
    assert type(cullpdb.trainlabel).__module__ == (np.__name__),'Data type must be numpy array'

    assert cullpdb.val_hot.shape == (256, 700),'Data shape must be (256x700)'
    assert cullpdb.valpssm.shape == (256, 700, 21),'Data shape must be (256x700x21)'
    assert cullpdb.vallabel.shape == (256, 700, 8),'Data shape must be (256x700x8)'

    assert type(cullpdb.val_hot).__module__ == (np.__name__),'Data type must be numpy array'
    assert type(cullpdb.valpssm).__module__ == (np.__name__),'Data type must be numpy array'
    assert type(cullpdb.vallabel).__module__ == (np.__name__),'Data type must be numpy array'

    assert cullpdb.is_filtered() == True, 'Dataset must be filtered'
    assert len(cullpdb) == 5278, 'Data length must be 5877'
    assert cullpdb.shape() == (5278, 700),'Data shape must be (5278x700)'
    assert cullpdb.url == "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"


    ### CullPDB 6133 unfiltered dataset tests ###
    cullpdb =  CullPDB(6133, filtered = False)

    assert cullpdb.train_hot.shape == (5600, 700),'Data shape must be (5600x700)'
    assert cullpdb.trainpssm.shape == (5600, 700, 21),'Data shape must be (5600x700)'
    assert cullpdb.trainlabel.shape == (5600, 700, 8),'Data shape must be (5600x700)'

    assert type(cullpdb.train_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.test_hot.shape == (272, 700),'Data shape must be ()'
    assert cullpdb.testpssm.shape == (272, 700, 21)
    assert cullpdb.testlabel.shape == (272, 700, 8)

    assert type(cullpdb.test_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.testpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.testlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.val_hot.shape == (256, 700)
    assert cullpdb.valpssm.shape == (256, 700, 21)
    assert cullpdb.vallabel.shape == (256, 700, 8)

    assert type(cullpdb.val_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.valpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.vallabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.is_filtered() == False
    assert len(cullpdb) == 5600
    assert cullpdb.shape() == (5600, 700)
    assert cullpdb.url == "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133.npy.gz"
#assert total size = 6133 (data_index + val + test)

    ### CullPDB 5926 filtered dataset tests ###
    cullpdb =  CullPDB(5926, filtered = True)

    assert cullpdb.train_hot.shape == (5109, 700)
    assert cullpdb.trainpssm.shape == (5109, 700, 21)
    assert cullpdb.trainlabel.shape == (5109, 700, 8)

    assert type(cullpdb.train_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.val_hot.shape == (256, 700)
    assert cullpdb.valpssm.shape == (256, 700, 21)
    assert cullpdb.vallabel.shape == (256, 700, 8)

    assert type(cullpdb.val_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.valpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.vallabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.is_filtered() == True
    assert len(cullpdb) == 5109
    assert cullpdb.shape() == (5109, 700)
    print('culldpb url', cullpdb.url)
    assert cullpdb.url == "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz"


    ### CullPDB 5926 unfiltered dataset tests ###
    cullpdb =  CullPDB(5926, filtered = False)

    assert cullpdb.train_hot.shape == (5430, 700)
    assert cullpdb.trainpssm.shape == (5430, 700, 21)
    assert cullpdb.trainlabel.shape == (5430, 700, 8)

    assert type(cullpdb.train_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.trainlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.val_hot.shape == (236, 700)
    assert cullpdb.valpssm.shape == (236, 700, 21)
    assert cullpdb.vallabel.shape == (236, 700, 8)

    assert type(cullpdb.val_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.valpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.vallabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.test_hot.shape == (255, 700)
    assert cullpdb.testpssm.shape == (255, 700, 21)
    assert cullpdb.testlabel.shape == (255, 700, 8)

    assert type(cullpdb.test_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.testpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cullpdb.testlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert cullpdb.is_filtered() == False
    assert len(cullpdb) == 5430
    assert cullpdb.shape() == (5430, 700)
    assert cullpdb.url == "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926.npy.gz"

    #get_onehot tests

def test_cb513():

    """
    Description:
        Testing CB513 test dataset class and all its associated functions
    Args:
        None
    Returns:
        Test passed or test failed

    """
    cb513 = CB513()

    assert cb513.test_hot.shape == (514,700)
    assert cb513.testpssm.shape == (514,700,21)
    assert cb513.testlabel.shape == (514,700,8)

    assert type(cb513.test_hot).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cb513.testpssm).__module__ == (np.__name__), 'Data type must be numpy array'
    assert type(cb513.testlabel).__module__ == (np.__name__), 'Data type must be numpy array'

    assert(cb513.shape() == (514,700))
    assert(len(cb513) == 514)
    assert (cb513.shape() == (514, 700))
    assert(cb513.url == "https://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz")

def test_casp10():

    """
    Description:
        Testing CASP10 test dataset class and all its associated functions
    Args:
        None
    Returns:
        Test passed or test failed

    """

    casp10 = CASP10()

    assert(casp10.casp10_data_test_hot.shape == (123, 700))
    assert(casp10.casp10_data_pssm.shape == (123, 700, 21))
    assert(casp10.test_labels.shape == (123, 700, 8))

    assert(type(casp10.casp10_data_test_hot).__module__ == (np.__name__)), 'Data type must be numpy array'
    assert(type(casp10.casp10_data_pssm).__module__ == (np.__name__)), 'Data type must be numpy array'
    assert(type(casp10.test_labels).__module__ == (np.__name__)), 'Data type must be numpy array'

    assert(casp10.shape() == (123,700))
    assert(len(casp10) == 123)
    assert (casp10.shape() == (123, 700))
#    assert(casp10.url == "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5")
    assert(casp10.url == "https://github.com/amckenna41/CDBLSTM_PSSP/raw/master/casp10.h5")

def test_casp11():

    """
    Description:
        Testing CASP11 test dataset class and all its associated functions
    Args:
        None
    Returns:
        Test passed or test failed

    """
    casp11 = CASP11()

    assert(casp11.casp11_data_test_hot.shape == (105, 700))
    assert(casp11.casp11_data_pssm.shape == (105, 700, 21))
    assert(casp11.test_labels.shape == (105, 700, 8))

    assert(type(casp11.casp11_data_test_hot).__module__ == (np.__name__)), 'Data type must be numpy array'
    assert(type(casp11.casp11_data_pssm).__module__ == (np.__name__)), 'Data type must be numpy array'
    assert(type(casp11.test_labels).__module__ == (np.__name__)), 'Data type must be numpy array'

    assert(casp11.shape() == (105,700))
    assert(len(casp11) == 105)
    assert (casp11.shape() == (105, 700))
    # assert(casp11.url == "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5")
    assert(casp11.url == "https://github.com/amckenna41/CDBLSTM_PSSP/raw/master/casp11.h5")

def run_dataset_tests():

    """
    Description:
        Run all dataset tests
    Args:
        None
    Returns:
        Test passed or test failed

    """
    test_dataset_url()
    print('URL tests passed')
    test_cullpdb()
    print('CullPDB training dataset tests passed')
    test_cb513()
    print('CB513 test dataset tests passed')
    test_casp10()
    print('CASP10 test dataset tests passed')
    test_casp11()
    print('CASP11 test dataset tests passed')
    print('')
    print('All Dataset tests passed')

if __name__ == '__main__':

    run_dataset_tests()
