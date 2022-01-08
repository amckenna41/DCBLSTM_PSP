################################################################################
##########                     Tests for datasets                     ##########
################################################################################

import os
import sys
sys.path.append('../')
import requests
import numpy as np
from psp.dataset import *
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

#Test Suite for CullPDB datasets
@unittest.skip("CullPDB tests are very memory intensive, skipping for now.")
class CullPDBTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Import CullPDB datasets. """
        self.cullpdb_5926_filt = CullPDB(filtered=True)
        self.cullpdb_5926_unfilt = CullPDB(filtered=False)
        self.cullpdb_6133_filt = CullPDB(type=6133,filtered=True)
        self.cullpdb_6133_unfilt = CullPDB(type=6133,filtered=False)

#1.)
    def test_input_params(self):
        """ Testing CullPDB input parameters and instance variables. """
        self.assertEqual(self.cullpdb_5926_filt.type, 5926, '')
        self.assertEqual(self.cullpdb_5926_filt.filtered, 1, '')
        self.assertEqual(self.cullpdb_5926_unfilt.type, 5926, '')
        self.assertEqual(self.cullpdb_5926_unfilt.filtered, 0, '')
        self.assertEqual(self.cullpdb_6133_filt.type, 6133, '')
        self.assertEqual(self.cullpdb_6133_filt.filtered, 1, '')
        self.assertEqual(self.cullpdb_6133_unfilt.type, 5926, '')
        self.assertEqual(self.cullpdb_5926_unfilt.filtered, 0, '')
        self.assertEqual(self.cullpdb_6133_filt.all_data,1, '')
        self.assertEqual(self.cullpdb_6133_filt.all_data,1, '')
        self.assertEqual(self.cullpdb_6133_unfilt.all_data,1, '')
        self.assertEqual(self.cullpdb_5926_unfilt.all_data,1, '')

        self.assertEqual(self.cullpdb_5926_filt.url,'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz','')
        self.assertEqual(self.cullpdb_5926_filt.train_path, 'cullpdb+profile_5926_filtered.npy.gz','')
        self.assertEqual(self.cullpdb_5926_unfilt.url,'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926.npy.gz','')
        self.assertEqual(self.cullpdb_5926_unfilt.train_path, 'cullpdb+profile_5926.npy.gz','')
        self.assertEqual(self.cullpdb_6133_filt.url,'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz','')
        self.assertEqual(self.cullpdb_6133_filt.train_path, 'cullpdb+profile_6133_filtered.npy.gz','')
        self.assertEqual(self.cullpdb_6133_unfilt.url,'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133.npy.gz','')
        self.assertEqual(self.cullpdb_6133_unfilt.train_path, 'cullpdb+profile_6133.npy.gz','')

#2.)
    def test_shapes(self):
        """ Testing object dataset and instance variable shapes. """
        self.assertEqual(self.cullpdb_5926_filt.shape, (5109, 700),'Data shape must be (5100x700)')
        self.assertEqual(self.cullpdb_5926_unfilt.shape, (5430, 700),'Data shape must be (5430x700)')
        self.assertEqual(self.cullpdb_6133_filt.shape, (5278, 700),'Data shape must be (5278x700)')
        self.assertEqual(self.cullpdb_6133_unfilt.shape, (5600, 700),'Data shape must be (5600x700)')

        self.assertEqual(self.cullpdb_5926_filt.train_hot.shape, (5278,700),'Data shape must be (5278x700)')
        self.assertEqual(self.cullpdb_5926_filt.trainpssm.shape, (5278,700,21),'Data shape must be (5278x700x21)')
        self.assertEqual(self.cullpdb_5926_filt.trainlabel.shape, (5278,700,8),'Data shape must be (5278x700x8)')

        self.assertEqual(self.cullpdb_5926_filt.val_hot.shape, (256,700),'Data shape must be (256x700)')
        self.assertEqual(self.cullpdb_5926_filt.valpssm.shape, (256,700,21),'Data shape must be (256x700x21)')
        self.assertEqual(self.cullpdb_5926_filt.vallabel.shape, (256,700,8),'Data shape must be (256x700x8)')

        self.assertTrue(type(self.cullpdb_5926_filt.train_hot).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cullpdb_5926_filt.trainpssm).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cullpdb_5926_filt.trainlabel).__module__ == (np.__name__),'Data type must be numpy array')

        self.assertTrue(type(self.cullpdb_5926_filt.val_hot).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cullpdb_5926_filt.valpssm).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cullpdb_5926_filt.vallabel).__module__ == (np.__name__),'Data type must be numpy array')

#3.)
    def test_length(self):
        """ Testing cullPDB dataset length. """
        self.assertEqual(len(self.cullpdb_5926_filt), 5600,'There should be 5600 proteins in dataset.')
        self.assertEqual(len(self.cullpdb_5926_unfilt), 5430,'There should be 5430 proteins in dataset.')
        self.assertEqual(len(self.cullpdb_6133_filt), 5278,'There should be 5278 proteins in dataset.')
        self.assertEqual(len(self.cullpdb_6133_unfilt), 5600,'There should be 5600 proteins in dataset.')

#4.)
    def test_is_filtered(self):
        """ Testing is_filtered function to test if dataset filtered or not. """
        self.assertTrue(self.cullpdb_5926_filt.is_filtered(), 'Dataset should be the filtered type.')
        self.assertFalse(self.cullpdb_5926_unfilt.is_filtered(), 'Dataset should be the unfiltered type.')
        self.assertTrue(self.cullpdb_6133_filt.is_filtered(), 'Dataset should be the filtered type.')
        self.assertFalse(self.cullpdb_6133_unfilt.is_filtered(), 'Dataset should be the unfiltered type.')

#5.)
    def test_data_labels(self):
        """ Testing get_data_labels func ensuring correct protein data returned. """
        self.assertTrue(type(self.cullpdb_5926_filt.get_data_labels(10)).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertEqual(self.cullpdb_5926_filt.get_data_labels(10).shape, (700,8),'Data shape must be (700x8)')

#6.)
    def test_size(self):
        """ Testing size function that returns the size of dataset in MB. """
        self.assertEqual(self.cullpdb_5926_filt.size(), '1633 MB', 'Size of CullPDB dataset should be around 1633 MB')
        self.assertEqual(self.cullpdb_5926_unfilt.size(), '1633 MB', 'Size of CullPDB dataset should be around 1633 MB')
        self.assertEqual(self.cullpdb_6133_filt.size(), '1633 MB', 'Size of CullPDB dataset should be around 1633 MB')
        self.assertEqual(self.cullpdb_6133_unfilt.size(), '1633 MB', 'Size of CullPDB dataset should be around 1633 MB')

#7.)
    @unittest.skip("Don't want to overload the FTP server each time tests are run.")
    def test_url(self):
        """ Testing CullPDB training dataset URL's. """
        r = requests.get(self.cullpdb_5926_filt.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content encoding should be gzip.')

        r = requests.get(self.cullpdb_5926_unfilt.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content encoding should be gzip.')

        r = requests.get(self.cullpdb_6133_filt.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content encoding should be gzip.')

        r = requests.get(self.cullpdb_6133_unfilt.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content encoding should be gzip.')

    @classmethod
    def tearDownClass(self):
        """ Deleting CullPDB datasets from memory. """
        del self.cullpdb_5926_filt
        del self.cullpdb_5926_unfilt
        del self.cullpdb_6133_filt
        del self.cullpdb_6133_unfilt

#Test Suite for CB513 dataset
class CB513Tests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Import CB513 dataset """
        self.cb513_ = CB513()

#1.)
    def test_input_params(self):
        """ Test CB513 input parameter values. """
        self.assertEqual(self.cb513_.url,"https://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz",'')
        self.assertEqual(self.cb513_.test_path, 'cb513+profile_split1.npy.gz','CB513 test path not correct.')

#2.)
    def test_shapes(self):
        """ Test CB513 dataset shapes. """
        self.assertEqual(self.cb513_.shape(), (514, 700),'Data shape should be (514,700).')
        self.assertEqual(self.cb513_.test_hot.shape, (514,700),'Data shape should be (514,700).')
        self.assertEqual(self.cb513_.test_pssm.shape, (514,700,21),'Data shape should be (514,700,21).')
        self.assertEqual(self.cb513_.test_labels.shape, (514,700,8),'Data shape should be (514,700,8).')

#3.)
    def test_datatypes(self):
        """ Test CB513 dataset types. """
        self.assertTrue(type(self.cb513_.test_hot).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cb513_.test_pssm).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.cb513_.test_labels).__module__ == (np.__name__),'Data type must be numpy array')

#4.)
    def test_dataset_length(self):
        """ Test CB513 dataset length. """
        self.assertEqual(len(self.cb513_), 514,'There should be 514 proteins in dataset.')

#5.)
    def test_size(self):
        """ Testing size function that returns the size of dataset in MB. """
        self.assertEqual(self.cb513_.size(), '156 MB', 'Size of CB513 dataset should be around 156 MB')

#6.)
    @unittest.skip("Don't want to overload the FTP server each time tests are run.")
    def test_url(self):
        """ Test CB513 URL's. """
        r = requests.get(self.cb513_.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content encoding should be of type gzip')

    @classmethod
    def tearDownClass(self):
        """ Delete CB513 dataset stored in memory. """
        del self.cb513_

#Test Suite for CASP10 dataset
class CASP10Tests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Importing CASP10 dataset. """
        self.casp10_ = CASP10()

#1.)
    def test_input_params(self):
        """ Test CASP10 dataset input parameters. """
        self.assertEqual(self.casp10_.url,"https://github.com/amckenna41/DCBLSTM_PSP/raw/master/data/casp10.h5",'')
        self.assertEqual(self.casp10_.test_path,'casp10.h5','Test path should be casp10.h5.')

#2.)
    def test_shapes(self):
        """ Test CASP10 dataset shapes. """
        self.assertEqual(self.casp10_.shape(), (123, 700),'CASP10 dataset shape should be (123x700).')
        self.assertEqual(self.casp10_.test_hot.shape, (123,700),'CASP10 data shape should be (123x700).')
        self.assertEqual(self.casp10_.test_pssm.shape, (123,700,21),'CASP10 data shape should be (123x700x21).')
        self.assertEqual(self.casp10_.test_labels.shape, (123,700,8),'CASP10 data shape should be (123x700x8).')

#3.)
    def test_datatypes(self):
        """ Test CASP10 dataset data types. """
        self.assertTrue(type(self.casp10_.test_hot).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.casp10_.test_pssm).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.casp10_.test_labels).__module__ == (np.__name__),'Data type must be numpy array')

#4.)
    def test_dataset_length(self):
        """ Test CASP10 dataset length. """
        self.assertEqual(len(self.casp10_), 123,'There should be 123 protein sequences in dataset.')

#5.)
    def test_size(self):
        """ Testing size function that returns the size of dataset in MB. """
        self.assertEqual(self.casp10_.size(), '2 MB', 'Size of CASP10 dataset should be around 2 MB')

#6.)
    @unittest.skip("Don't want to overload the FTP server each time tests are run.")
    def test_url(self):
        """ Test CASP10 URL. """
        r = requests.get(self.casp10_.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content-encoding should be gzip.')

    @classmethod
    def tearDownClass(self):
        """ Deleting CASP10 dataset from memory. """
        del self.casp10_

#Test Suite for CASP11 dataset
class CASP11Tests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Importing CASP11 dataset. """
        self.casp11_ = CASP11()

#1.)
    def test_input_params(self):
        """ Test CASP11 dataset input parameters. """
        self.assertEqual(self.casp11_.url,"https://github.com/amckenna41/DCBLSTM_PSP/raw/master/data/casp11.h5",'')
        self.assertEqual(self.casp11_.test_path,'casp11.h5','Dataset test path should be casp11.h5.')

#2.)
    def test_shapes(self):
        """ Test CASP10 dataset shapes. """
        self.assertEqual(self.casp11_.shape(), (105, 700),'Dataset shape should be (105x700).')
        self.assertEqual(self.casp11_.test_hot.shape, (105,700),'Data shape should be (105x700).')
        self.assertEqual(self.casp11_.test_pssm.shape, (105,700,21),'Data shape should be (105x700x21).')
        self.assertEqual(self.casp11_.test_labels.shape, (105,700,8),'Data shape should be (105x700x8).')

#3.)
    def test_datatypes(self):
        """ Test CASP11 dataset data types. """
        self.assertTrue(type(self.casp11_.test_hot).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.casp11_.test_pssm).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(type(self.casp11_.test_labels).__module__ == (np.__name__),'Data type must be numpy array')

#4.)
    def test_dataset_length(self):
        """ Test CASP10 dataset length. """
        self.assertEqual(len(self.casp11_), 105,'There should be 105 protein sequences in dataset.')

#5.)
    def test_size(self):
        """ Testing size function that returns the size of dataset in MB. """
        self.assertEqual(self.casp11_.size(), '2 MB', 'Size of CASP11 dataset should be around 2 MB')

#6.)
    @unittest.skip("Don't want to overload the FTP server each time tests are run.")
    def test_url(self):
        """ Test CASP11 URL. """
        r = requests.get(self.casp11_.url, allow_redirects = True)
        self.assertTrue(r.status_code == 200, 'Error getting URL, status code: {}'.format(r.status_code))
        self.assertEqual(r.headers['Content-Type'],'application/x-gzip','Content-type should be application/x-gzip.')
        self.assertEqual(r.headers['Content-Encoding'],'gzip','Content-encoding should be gzip.')

    @classmethod
    def tearDownClass(self):
        """ Deleting CASP11 dataset from memory. """
        del self.casp11_

if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)
