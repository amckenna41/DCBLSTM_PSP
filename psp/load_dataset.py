################################################################################
###################### Loading training and test datasets ######################
################################################################################

#importing libraries and dependancies
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import gzip
import h5py
import os, sys
from pathlib import Path
try:
    from _globals import *
except:
    from . _globals import *
import requests
import shutil

class CullPDB():
    """
    Description:
        CullPDB class creates instance of the CullPDB 5926/6133 training datasets.
    Args:
        :type (str): specifying what CullPDB dataset to use for training, 5926 or 6133.
        :all_data (float): The proportion of the training data to use, must be in range 0.5 to 1.0, default: 1.0,
            used for testing purposes to not have to load in full dataset each time.
        :filtered (bool): What cullPDB training dataset to use, filtered or unfiltered. Filtered
            dataset has protein overlap with CB513 test dataset removed, unfiltered does not, default = True.
    Returns:
        CullPDB training dataset object
    """
    def __init__(self, type="5926", filtered=True, all_data=1, save_dir=DATA_DIR):

        TRAIN_PATH_FILTERED_6133 = 'cullpdb+profile_6133_filtered.npy.gz'
        TRAIN_PATH_UNFILTERED_6133 = 'cullpdb+profile_6133.npy.gz'

        TRAIN_FILTERED_6133_URL = "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
        TRAIN_UNFILTERED_6133_URL = "https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133.npy.gz"

        TRAIN_PATH_FILTERED_5926 = 'cullpdb+profile_5926_filtered.npy.gz'
        TRAIN_PATH_UNFILTERED_5926 = 'cullpdb+profile_5926.npy.gz'

        TRAIN_FILTERED_5926_URL = 'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz'
        TRAIN_UNFILTERED_5926_URL = 'https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926.npy.gz'

        self.filtered = filtered
        self.type = type
        self.all_data = all_data

        #if dataset not 5926 or 6133 then set to 5926 default.
        if (self.type!="6133" and self.type!="5926"):
            self.type="5926"

        #set training path and url class variables
        if (self.type == "6133"):
            if (self.filtered):
                self.train_path = TRAIN_PATH_FILTERED_6133
                self.url = TRAIN_FILTERED_6133_URL
            else:
                self.train_path = TRAIN_PATH_UNFILTERED_6133
                self.url = TRAIN_UNFILTERED_6133_URL
        else:
            if (self.filtered):
                self.train_path = TRAIN_PATH_FILTERED_5926
                self.url = TRAIN_FILTERED_5926_URL
            else:
                self.train_path = TRAIN_PATH_UNFILTERED_5926
                self.url = TRAIN_UNFILTERED_5926_URL

        #download dataset if not already in current directory
        if not (os.path.isfile(os.path.join(DATA_DIR, self.train_path[:-3]))):
            self.download_cullpdb()

        #if all_data input parameter not between 0 and 1 then set to 1
        if (self.all_data > 1) or (self.all_data < 0):
            self.all_data = 1

        #load in dataset
        self.load_cullpdb()

    def load_cullpdb(self):
        """
        Description:
            Load CullPDB 5926/6133 training dataset from data directory.
        Args:
            :self (CullPDB object): instance of CullPDB class.
        Returns:
            None
        """
        print("\nLoading CullPDB {} training dataset (filtered: {})...\n".format(self.type, self.filtered))
        #load dataset
        data = np.load(os.path.join(DATA_DIR, self.train_path[:-3]))

        #reshape dataset
        data = np.reshape(data, (-1, 700, 57))
        #sequence feature
        datahot = data[:, :, 0:21]
        #profile features
        datapssm = data[:, :, 35:56]
        #secondary struture labels
        labels = data[:, :, 22:30]

        # shuffle data
        num_seqs, seqlen, feature_dim = np.shape(data)
        num_classes = labels.shape[2]
        seq_index = np.arange(0, num_seqs)#
        np.random.shuffle(seq_index)

        if self.type == "6133":

            if (self.filtered):
                data_index = 5278
                # val_index = 5534 - 5278

                valhot = datahot[seq_index[5278:5534]] #21
                vallabel = labels[seq_index[5278:5534]] #8
                valpssm = datapssm[seq_index[5278:5534]] # 21

                #one-hot encoding of val labels
                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

            else:
                # calculate the indexes for each dimension based on all_data input parameter
                data_index = 5600
                test_index = 272
                val_index = 256

                test_labels = datahot[seq_index[5605:5877]]
                test_labels = labels[seq_index[5605:5877]]
                test_pssm = datapssm[seq_index[5605:5877]]

                #one-hot encoding of test labels
                test_hot = np.ones((test_labels.shape[0], test_labels.shape[1])) #target array
                self.test_hot = get_onehot(test_labels, test_hot)

                valhot = datahot[seq_index[5877:6133]] #21
                vallabel = labels[seq_index[5877:6133]] #8
                valpssm = datapssm[seq_index[5877:6133]] # 21

                #one-hot encoding of val labels
                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)
        else:

            if (self.filtered):
                data_index = 5365 - 256
                val_index = 256

                valhot = datahot[seq_index[5109:5365]] #21
                vallabel = labels[seq_index[5109:5365]] #8
                valpssm = datapssm[seq_index[5109:5365]] # 21

                #one-hot encoding of val labels
                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)
            else:
                data_index = 5430
                val_index = 5926-5690

                valhot = datahot[seq_index[5690:5926]] #21
                vallabel = labels[seq_index[5690:5926]] #8
                valpssm = datapssm[seq_index[5690:5926]] # 21

                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

                test_labels = datahot[seq_index[5435:5690]]
                test_labels = labels[seq_index[5435:5690]]
                test_pssm = datapssm[seq_index[5435:5690]]

                #one-hot encoding of test labels
                test_hot = np.ones((test_labels.shape[0], test_labels.shape[1])) #target array
                self.test_hot = get_onehot(test_labels, test_hot)  #test_hot (target) not crrated

        #extract training data from arrays
        trainhot = datahot[seq_index[:data_index]]
        trainlabel = labels[seq_index[:data_index]]
        trainpssm = datapssm[seq_index[:data_index]]

        #one-hot encoding of train labels
        train_hot = np.ones((trainhot.shape[0], trainhot.shape[1])) #target array
        train_hot = get_onehot(trainhot, train_hot)

        #delete training data from memory
        del data

        #set cullpdb class variables
        self.train_hot = train_hot
        self.trainpssm = trainpssm
        self.trainlabel = trainlabel
        self.val_hot = val_hot
        self.valpssm = valpssm
        self.vallabel = vallabel
        if not(self.filtered):
            self.test_hot = test_hot
            self.test_labels = test_labels
            self.test_pssm = test_pssm

    def download_cullpdb(self):
        """
        Description:
            Download Cullpdb dataset from Princeton URL and store locally in data directory.
        Args:
            :self (CullPDB object): instance of CullPDB class.
        Returns:
            None.
        """
        #get data from url
        r = requests.get(self.url, allow_redirects = True)
        r.raise_for_status()    #if response status code != 200

        #open local file for writing
        open(os.path.join(DATA_DIR, self.train_path), 'wb').write(r.content)

        #unzip dataset and save to directory
        with gzip.open(os.path.join(DATA_DIR, self.train_path), 'rb') as f_in:
            with open(os.path.join(DATA_DIR, self.train_path[:-3]), 'wb') as f_out:
                shutil.copyfile(os.path.join(DATA_DIR, self.train_path),
                    os.path.join(DATA_DIR, self.train_path[:-3]))

        #remove unrequired zipped version of dataset
        os.remove(os.path.join(DATA_DIR, self.train_path))

        print('Dataset downloaded successfully - stored in {} of size {} \n'.format(
            os.path.join(DATA_DIR, self.train_path[:-3]),self.dataset_size()))

    def __len__(self):
        """ Get number of proteins in CullPDB dataset - length of the 1st dimension. """
        return (self.train_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CullPDB object. """
        return ('CullPDB {} Training datatset - filtered: {}. \
            Shape of dataset: {}.'.format(self.type,self.filtered, self.shape()))

    def is_filtered(self):
        """ Is current CullPDB class object using the filtered or unfiltered dataset? """
        return self.filtered

    def shape(self):
        """ Output shape of CullPDB object. """
        return self.train_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CullPDB dataset specified by protein index argument. """
        return self.trainlabel[protein_index,:,:]

    def dataset_size(self):
        """ Get size of CullPDB dataset """
        return str(round((os.path.getsize(os.path.join(DATA_DIR,self.train_path[:-3])))/(1024*1024))) + ' MB'

class CB513():
    """
    Description:
        CB513 class creates instance of the CB513 test dataset.
    Args:
        None
    Returns:
        CB513 test dataset object
    """
    def __init__(self, save_dir=DATA_DIR):

        CB513_PATH = 'cb513+profile_split1.npy.gz'
        CB513_URL = "https://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"

        self.url = CB513_URL
        self.test_path = CB513_PATH

        print("\nLoading CB513 test dataset ...\n")

        #download dataset if not already in data directory
        if not (os.path.isfile(os.path.join(DATA_DIR, self.test_path[:-3]))):
            self.download_cb513()

        #load in dataset
        self.load_cb513()

    def load_cb513(self):
        """
        Description:
            Load CB513 test dataset from data directory.
        Args:
            :self (CB513 object): instance of CB513 class.
        Returns:
            None
        """
        #load test dataset
        CB513 = np.load(os.path.join(DATA_DIR,self.test_path[:-3]))

        #reshape dataset
        CB513= np.reshape(CB513,(-1,700,57))
        #sequence feature
        test_labels=CB513[:, :, 0:21]
        #profile feature
        test_pssm=CB513[:, :, 35:56]
        #secondary struture label
        test_labels = CB513[:, :, 22:30]

        test_labels = test_labels[:514]
        test_pssm = test_pssm[:514]
        test_labels = test_labels[:514]

        #convert to one-hot encoding array
        test_hot = np.ones((test_labels.shape[0], test_labels.shape[1]))
        test_hot = get_onehot(test_labels, test_hot)

        #delete test data from memory
        del CB513

        #set cb513 class variables
        self.test_hot = test_hot
        self.test_pssm = test_pssm
        self.test_labels = test_labels

    def download_cb513(self):
        """
        Description:
            Download CB513 dataset from Princeton URL and store locally in data directory.
        Args:
            :self (CB513 object): instance of CB513 class.
        Returns:
            None.
        """
        #get data from url
        r = requests.get(self.url, allow_redirects = True)
        r.raise_for_status()    #if response status code != 200

        #open local file for writing
        open(os.path.join(DATA_DIR, self.test_path), 'wb').write(r.content)

        #unzip test dataset
        with gzip.open(os.path.join(DATA_DIR, self.test_path), 'rb') as f_in:
            with open(os.path.join(DATA_DIR, self.test_path[:-3]), 'wb') as f_out:
                shutil.copyfile(os.path.join(DATA_DIR, self.test_path),
                    os.path.join(DATA_DIR, self.test_path[:-3]))

        #remove unrequired zipped version of dataset
        os.remove(os.path.join(DATA_DIR, self.test_path))

        print('Dataset downloaded successfully - stored in {} of size {} \n'.format(
            os.path.join(DATA_DIR, self.test_path[:-3]),self.dataset_size()))

    def __len__(self):
        """ Get number of proteins in CB513 dataset - length of the 1st dimension. """
        return (self.test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CB513 object. """
        return ('CB513 Test datatset. Shape of dataset: {}.'.format(self.shape()))

    def shape(self):
        """ Output shape of CB513 object. """
        return self.test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CB513 dataset specified by protein index argument. """
        return self.test_labelsel[protein_index,:,:]

    def dataset_size(self):
        """ Get size of CB513 dataset """
        return str(round((os.path.getsize(os.path.join(DATA_DIR,self.test_path[:-3])))/(1024*1024))) + ' MB'

class CASP10():
    """
    Description:
        CASP10 class creates instance of the CASP10 test dataset.
    Args:
        None
    Returns:
        CASP10 test dataset object.
    """
    def __init__(self):

        CASP10_PATH = 'casp10.h5'
        CASP10_URL = "https://github.com/amckenna41/DCBLSTM_PSP/raw/master/data/casp10.h5"
        self.url = CASP10_URL
        self.test_path = CASP10_PATH

        print("\nLoading CASP10 dataset...\n")

        #download dataset if not already in data directory
        if not (os.path.isfile(os.path.join(DATA_DIR, self.test_path))):
            self.download_casp10()

        #load in dataset
        self.load_casp10()

    def load_casp10(self):
        """
        Description:
            Load CASP10 test dataset from data directory.
        Args:
            :self (CASP10 object): instance of CASP10 class.
        Returns:
            None
        """
        #load casp10 dataset
        casp10_data = h5py.File(os.path.join(DATA_DIR, self.test_path),'r')

        #load protein sequence and profile feature data
        data_hot = casp10_data['features'][:, :, 0:21]
        test_pssm = casp10_data['features'][:, :, 21:42]

        #load protein label data
        test_labels = casp10_data['labels'][:, :, 0:8]

        #crete one hot encoding vector for casp10 labels
        test_hot = np.ones((data_hot.shape[0], data_hot.shape[1]))
        test_hot = get_onehot(data_hot, test_hot)

        #delete test data from memory
        del casp10_data

        #set casp10 class variables
        self.test_hot = test_hot
        self.test_pssm = test_pssm
        self.test_labels = test_labels

    def download_casp10(self):
        """
        Description:
            Download CASP10 dataset from repository and store locally in data directory.
        Args:
            :self (CASP10 object): instance of CASP10 class.
        Returns:
            None.
        """
        #get data from url
        r = requests.get(self.url, allow_redirects = True)
        r.raise_for_status() #if response status code != 200

        #open local file for writing
        open(os.path.join(DATA_DIR, self.test_path), 'wb').write(r.content)

        print('Dataset downloaded successfully - stored in {} of size {} \n'.format(
            os.path.join(DATA_DIR, self.test_path),self.dataset_size()))

    def __len__(self):
        """ Get number of proteins in CASP10 dataset - length of the 1st dimension. """
        return (self.test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CASP10 object. """
        return ('CASP10 Test datatset. Shape of dataset: {}.'.format(self.shape()))

    def shape(self):
        """ Output shape of CASP10 object. """
        return self.test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP10 dataset specified by protein index argument. """
        return self.test_labels[protein_index,:,:]

    def dataset_size(self):
        """ Get size of CASP10 dataset. """
        return str(round((os.path.getsize(os.path.join(DATA_DIR,self.test_path)))/(1024*1024))) + ' MB'

class CASP11():
    """
    Description:
        CASP11 class creates instance of the CASP11 test dataset.
    Args:
        None
    Returns:
        CASP11 test dataset object.
    """
    def __init__(self):

        CASP11_PATH = 'casp11.h5'
        CASP11_URL = "https://github.com/amckenna41/DCBLSTM_PSP/raw/master/data/casp11.h5"

        self.url = CASP11_URL
        self.test_path = CASP11_PATH

        print("\nLoading CASP11 dataset...\n")

        #download dataset if not already in current directory
        if not (os.path.isfile(os.path.join(DATA_DIR, self.test_path))):
            self.download_casp11()

        #load in dataset
        self.load_casp11()

    def load_casp11(self):
        """
        Description:
            Load CASP11 test dataset from data directory.
        Args:
            :self (CASP11 object): instance of CASP11 class.
        Returns:
            None
        """
        #load casp11 dataset
        casp11_data = h5py.File(os.path.join(DATA_DIR, self.test_path),'r')

        #load protein sequence feature data
        data_hot = casp11_data['features'][:,:,0:21]
        # load profile feature data
        test_pssm = casp11_data['features'][:,:,21:42]
        #load protein label feature data
        test_labels = casp11_data['labels'][:,:,0:8]

        #convert to one-hot encoding array
        test_hot = np.ones((data_hot.shape[0], data_hot.shape[1]))
        test_hot = get_onehot(data_hot, test_hot)

        #delete test data from ram
        del casp11_data

        #set casp11 class variables
        self.test_hot = test_hot
        self.test_pssm = test_pssm
        self.test_labels = test_labels

    def download_casp11(self):
        """
        Description:
            Download CASP11 dataset from repository and store locally in data directory.
        Args:
            :self (CASP11 object): instance of CASP11 class.
        Returns:
            None.
        """
        #get data from url
        r = requests.get(self.url, allow_redirects = True)
        r.raise_for_status()

        #open local file for writing
        open(os.path.join(DATA_DIR,self.test_path), 'wb').write(r.content)

        print('Dataset downloaded successfully - stored in {} of size {} \n'.format(
            os.path.join(DATA_DIR, self.test_path),self.dataset_size()))

    def __len__(self):
        """ Get number of proteins in CASP11 dataset - length of the 1st dimension. """
        return (self.test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CASP11 object. """
        return ('CASP11 Test datatset. Shape of dataset: {}.'.format(self.shape()))

    def shape(self):
        """ Output shape of CASP11 object. """
        return self.test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP11 dataset specified by protein index argument. """
        return self.test_labels[protein_index,:,:]

    def dataset_size(self):
        """ Get size of CASP11 dataset. """
        return str(round((os.path.getsize(os.path.join(DATA_DIR,self.test_path)))/(1024*1024))) + ' MB'

def get_onehot(source_array, target_array):
    """
    Description:
        Convert protein structure labels into one-hot encoding vector.
    Args:
        :source_array (np.array): array of class labels to convert into one-hot vector.
        :target_array (np.array): array of class labels converted into one-hot encoding.
    Returns:
        :target_array (np.array): array of class labels converted into one-hot encoding vector.
    """
    target_array = np.ones((source_array.shape[0], source_array.shape[1]))
    for i in range(source_array.shape[0]):
        for j in range(source_array.shape[1]):
            if np.sum(source_array[i,j,:]) != 0:
                target_array[i,j] = np.argmax(source_array[i,j,:])

    return target_array

def download_all_data():
    """
    Description:
        Download all datatsets, training and test, used in project.
    Args:
        None
    Returns:
        None
    """
    cullpdb_6133_filt = CullPDB()
    cullpdb_6133_unfilt = CullPDB(filtered=False)
    cullpdb_5926_filt = CullPDB(type="5926")
    cullpdb_5926_unfilt = CullPDB(type="5926", filtered=False)
    cb513 = CB513()
    casp10 = CASP10()
    casp11 = CASP11()
