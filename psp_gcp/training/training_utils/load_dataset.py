###Downloading the training and test datasets and uploading them to a GCP Storage Bucket
#Datasets downloaded to local psp_gcp directory as they are required when the GCP job is packaged
#and sent to GCP Ai-Platform

#importing libraries and dependancies
import numpy as np
import gzip
import h5py
import os
import requests
import shutil
import argparse
import training.training_utils.gcp_utils as utils
from training.training_utils.globals import *
from sklearn.preprocessing import OneHotEncoder


class CullPDB(object):

    """
        Description:
            CullPDB class creates instance of the training dataset.

        Args:
            all_data (float): The proportion of the training data to use, must be in range 0.5 to 1.0, default: 1.0
            filtered (bool): What PDB training dataset to use, filtered or unfiltered, default = True.

        Returns:
            CullPDB training dataset object
    """

    def __init__(self, type= 5926, filtered = True):

        self.filtered = filtered
        self.type = int(type)

        assert (self.type == 6133 or self.type ==5926), f'training datatset must be of type 6133 or 5926 but got {self.type}'

        if (self.type == 6133):
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
        if not (os.path.isfile(self.train_path[:-3])):
            self.get_cullpdb()

        #load in dataset
        self.load_cullpdb()

    def load_cullpdb(self):
        """ Load CullPDB training dataset into memory """

        print("Loading CullPDB {} training dataset (filtered: {})...\n".format(self.type, self.filtered))

        #load dataset
        data = np.load(self.train_path[:-3])

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

        if self.type == 6133:

            if (self.filtered):
                data_index = 5278
                # val_index = 5534 - 5278

                valhot = datahot[seq_index[5278:5534]] #21
                vallabel = labels[seq_index[5278:5534]] #8
                valpssm = datapssm[seq_index[5278:5534]] # 21

                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

            else:
                # #calculate the indexes for each dimension based on all_data input parameter
                data_index = 5600
                test_index = 272
                val_index = 256

                testhot = datahot[seq_index[5605:5877]]
                testlabel = labels[seq_index[5605:5877]]
                testpssm = datapssm[seq_index[5605:5877]]

                test_hot = np.ones((testhot.shape[0], testhot.shape[1])) #target array
                self.test_hot = get_onehot(testhot, test_hot)

                valhot = datahot[seq_index[5877:6133]] #21
                vallabel = labels[seq_index[5877:6133]] #8
                valpssm = datapssm[seq_index[5877:6133]] # 21

                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

        else:

            if (self.filtered):
                data_index = 5365 - 256
                val_index = 256

                # val_index = int(236)
                valhot = datahot[seq_index[5109:5365]] #21
                vallabel = labels[seq_index[5109:5365]] #8
                valpssm = datapssm[seq_index[5109:5365]] # 21

                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

            else:
                data_index = 5430
                val_index = 5926-5690

                # val_index = int(236)
                valhot = datahot[seq_index[5690:5926]] #21
                vallabel = labels[seq_index[5690:5926]] #8
                valpssm = datapssm[seq_index[5690:5926]] # 21

                val_hot = np.ones((valhot.shape[0], valhot.shape[1])) #target array
                val_hot = get_onehot(valhot, val_hot)

                testhot = datahot[seq_index[5435:5690]]
                testlabel = labels[seq_index[5435:5690]]
                testpssm = datapssm[seq_index[5435:5690]]

                test_hot = np.ones((testhot.shape[0], testhot.shape[1])) #target array
                self.test_hot = get_onehot(testhot, test_hot)  #test_hot (target) not crrated


        print('data index:', data_index)
        print('seqindex shaoe:', seq_index.shape)

        trainhot = datahot[seq_index[:data_index]]
        trainlabel = labels[seq_index[:data_index]]
        trainpssm = datapssm[seq_index[:data_index]]
        print('trainpssm shaoe:', trainpssm.shape)
        print('trainlabel shaoe:', trainlabel.shape)
        print('trainhot shaoe:', trainhot.shape)

        train_hot = np.ones((trainhot.shape[0], trainhot.shape[1])) #target array
        train_hot = get_onehot(trainhot, train_hot)

        #delete training data from ram
        del data

        self.train_hot = train_hot
        self.trainpssm = trainpssm
        self.trainlabel = trainlabel
        self.val_hot = val_hot
        self.valpssm = valpssm
        self.vallabel = vallabel
        if not(self.filtered):
            self.test_hot = test_hot
            self.testlabel = testlabel
            self.testpssm = testpssm

    def get_cullpdb(self):
        """ Download Cullpdb dataset from Princeton URL and store locally in data directory """

        try:
            print(self.train_path[:-3])

            if not (os.path.isfile(self.train_path[:-3])):

                r = requests.get(self.url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = DATA_DIR
                source_path = self.train_path
                destination_path = self.train_path[:-3]

                open(source_path, 'wb').write(r.content)

                print('Exporting CullPDB {} {} datatset....'.format(self.type, self.filtered))
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                os.remove(source_path)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(destination_path,self.dataset_size()))

            else:
                #dataset already present
                print("CullPDB {} already present...".format(self.type))

        except OSError:
            print('Error downloading and exporting training dataset\n')

    def __len__(self):
        """ Get number of proteins in CullPDB dataset - length of the 1st dimension """
        return (self.train_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CullPDB object """
        return ('CullPDB {} Training datatset - filtered: {}. Shape of dataset: {}'.format(self.type,self.filtered, self.shape()))

    def is_filtered(self):
        """ Is current CullPDB class object using the filtered or unfiltered dataset """
        return self.filtered

    def shape(self):
        """ Output shape of CullPDB object """
        return self.train_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CullPDB dataset specified by protein index argument """
        labels = self.trainlabel[protein_index,:,:]
        return labels

    def dataset_size(self):
        """ Get size of CullPDB dataset """
        return str(round((os.path.getsize(self.train_path))/(1024*1024))) + ' MB'


    # def __len__(self):
    #     """      """
    #     return get_num_features(self)
    #
    # def __doc__(self):
    #     pass
    #
    # def __repr__(self):
    #     pass
    #
    # def __sizeof__(self):
    #     pass


class CB513(object):

    """
        Description:
            CB513 class creates instance of the CB513 test dataset.

        Args:
            None

        Returns:
            CB513 test dataset object
    """

    def __init__(self):

        self.url = CB513_URL
        self.test_path = CB513_PATH

        print("Loading test dataset (CB513)...\n")

        #download dataset if not already in current directory
        # # if not (os.path.isfile(TEST_PATH)):
        # if not (os.path.isfile(os.path.join(DATA_DIR, self.test_path[:-3]))):
        #     self.get_cb513()

        #download dataset if not already in current directory
        if not (os.path.isfile(self.test_path[:-3])):
            self.get_cb513()

        #load in dataset
        self.load_cb513()

    def load_cb513(self):
        """ Load CB513 test dataset into memory """

        #load test dataset
        CB513 = np.load(self.test_path[:-3])

        #reshape dataset
        CB513= np.reshape(CB513,(-1,700,57))
        #sequence feature
        testhot=CB513[:, :, 0:21]
        #profile feature
        testpssm=CB513[:, :, 35:56]
        #secondary struture label
        testlabel = CB513[:, :, 22:30]

        testhot = testhot[:514]
        testpssm = testpssm[:514]
        testlabel = testlabel[:514]

        #convert to one-hot array
        test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
        test_hot = get_onehot(testhot, test_hot)

        #delete test data from ram
        del CB513

        self.test_hot = test_hot
        self.testpssm = testpssm
        self.testlabel = testlabel

    def get_cb513(self):
        """ Download CASP11 dataset from Princeton URL and store locally in data directory """

        try:
            if not (os.path.isfile(self.test_path[:-3])):

                r = requests.get(self.url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = DATA_DIR
                source_path = self.test_path
                destination_path = self.test_path[:-3]

                open(source_path, 'wb').write(r.content)

                print('Exporting CB513 datatset....')
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                os.remove(source_path)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(destination_path,self.dataset_size()))

            else:

                #dataset already present
                print(str(self) + " already present...")

        except OSError:
            print('Error downloading and exporting dataset\n')

    def __len__(self):
        """ Get number of proteins in CB513 dataset - length of the 1st dimension """
        return (self.test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CB513 object """
        return ('CB513 Test datatset. Shape of dataset: {} '.format(self.shape()))

    def shape(self):
        """ Output shape of CB513 object """
        return self.test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CB513 dataset specified by protein index argument """
        labels = self.testlabel[protein_index,:,:]
        return labels

    def dataset_size(self):
        """ Get size of CB513 dataset """
        return str(round((os.path.getsize(self.test_path))/(1024*1024))) + ' MB'

    # def __len__(self):
    #     """      """
    #     return get_num_features(self)
    #
    # def __doc__(self):
    #     pass
    #
    # def __repr__(self):
    #     pass
    #
    # def __sizeof__(self):
    #     pass


class CASP10(object):

    """
        Description:
            CASP10 class creates instance of the CASP10 test dataset.

        Args:
            None

        Returns:
            CASP10 test dataset object
    """

    def __init__(self):

        self.url = CASP10_URL
        self.test_path = CASP10_PATH

        print("Loading CASP10 dataset...\n")

        #download dataset if not already in current directory
        if not (os.path.isfile(self.test_path[:-3])):
            self.get_casp10()

        #load in dataset
        self.load_casp10()

    def load_casp10(self):
        """ Load CASP10 test dataset into memory """

        #load casp10 dataset
        casp10_data = h5py.File(self.test_path,'r')

        #load protein sequence and profile feature data
        casp10_data_hot = casp10_data['features'][:, :, 0:21]
        casp10_data_pssm = casp10_data['features'][:, :, 21:42]

        #load protein label data
        test_labels = casp10_data['labels'][:, :, 0:8]

        #create new protein sequence feature, set values to max value if if value!=0 ?
        casp10_data_test_hot = np.ones((casp10_data_hot.shape[0], casp10_data_hot.shape[1]))
        casp10_data_test_hot = get_onehot(casp10_data_hot, casp10_data_test_hot)

        print('CASP10 dataset loaded...\n')

        #delete test data from ram
        del casp10_data

        self.casp10_data_test_hot = casp10_data_test_hot
        self.casp10_data_pssm = casp10_data_pssm
        self.test_labels = test_labels


    def get_casp10(self):
        """ Download CASP10 dataset from repository and store locally in data directory """

        try:

            if not (os.path.isfile(self.test_path)):

                r = requests.get(self.url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = DATA_DIR
                source_path = self.test_path

                open(source_path, 'wb').write(r.content)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(source_path,self.dataset_size()))

            else:
                #dataset already present
                print(str(self) + " already present...")

        except OSError:
            print('Error downloading and exporting dataset\n')

    def __len__(self):
        """ Get number of proteins in CASP10 dataset - length of the 1st dimension """
        return (self.casp10_data_test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CASP10 object """
        return ('CASP10 Test datatset. Shape of dataset: {} '.format(self.shape()))

    def shape(self):
        """ Output shape of CASP10 object """
        return self.casp10_data_test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP10 dataset specified by protein index argument """
        labels = self.test_labels[protein_index,:,:]
        return labels

    def dataset_size(self):
        """ Get size of CASP10 dataset """
        return str(round((os.path.getsize(self.test_path))/(1024*1024))) + ' MB'


class CASP11(object):

    """
        Description:
            CASP11 class creates instance of the CASP11 test dataset.

        Args:
            None

        Returns:
            CASP11 test dataset object
    """

    def __init__(self):

        self.url = CASP11_URL
        self.test_path = CASP11_PATH

        print("Loading CASP11 dataset...\n")

        #download dataset if not already in current directory
        if not (os.path.isfile(self.test_path[:-3])):
            self.get_casp11()

        #load in dataset
        self.load_casp11()

    def load_casp11(self):
        """ Load CASP11 test dataset into memory """

        #load casp11 dataset
        casp11_data = h5py.File(self.test_path,'r')

        #load protein sequence and profile feature data
        casp11_data_hot = casp11_data['features'][:,:,0:21]
        casp11_data_pssm = casp11_data['features'][:,:,21:42]
        #load protein label data
        test_labels = casp11_data['labels'][:,:,0:8]

        #create new protein sequence feature, set values to max value if if value!=0 ?
        casp11_data_test_hot = np.ones((casp11_data_hot.shape[0], casp11_data_hot.shape[1]))
        casp11_data_test_hot = get_onehot(casp11_data_hot, casp11_data_test_hot)

        print('CASP11 dataset loaded...\n')

        #delete test data from ram
        del casp11_data

        self.casp11_data_test_hot = casp11_data_test_hot
        self.casp11_data_pssm = casp11_data_pssm
        self.test_labels = test_labels

    def get_casp11(self):
        """ Download CASP11 dataset from repository and store locally in data directory """

        try:
            if not (os.path.isfile(self.test_path)):

                r = requests.get(self.url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = DATA_DIR
                source_path = self.test_path

                open(source_path, 'wb').write(r.content)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(source_path,self.dataset_size()))

            else:
                #dataset already present
                print(str(self) + " already present...")

        except OSError:
            print('Error downloading and exporting dataset\n')


    def __len__(self):
        """ Get number of proteins in CASP11 dataset - length of the 1st dimension """
        return (self.casp11_data_test_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CASP11 object """
        return ('CASP11 Test datatset. Shape of dataset: {} '.format(self.shape()))

    def shape(self):
        """ Output shape of CASP11 object """
        return self.casp11_data_test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP11 dataset specified by protein index argument """
        labels = self.test_labels[protein_index,:,:]
        return labels

    def dataset_size(self):
        """ Get size of CASP11 dataset """
        return str(round((os.path.getsize(self.test_path))/(1024*1024))) + ' MB'


def get_onehot(source_array, target_array):
    """
    Description:
        Convert protein structure labels into one-hot encoding vector

    Args:
        source_array (np.array): array of class labels to convert into one-hot vector
        target_array (np.array): array of class labels converted into one-hot encoding

    Returns:
        target_array (np.array): array of class labels converted into one-hot encoding vector
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
            Download all datatsets - training and test - used in project
        Args:
            None
        Returns:
            None
    """

    cullpdb_5926_filt = CullPDB(type=5926)
    cullpdb_5926_unfilt = CullPDB(type=5926, filtered=False)
    cullpdb_6133_filt = CullPDB()
    cullpdb_6133_unfilt = CullPDB(filtered=False)
    cb513 = CB513()
    casp10 = CASP10()
    casp11 = CASP11()
