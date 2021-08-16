##########################################
### Loading training and test datasets ###
##########################################

#importing libraries and dependancies
import numpy as np
import gzip
import h5py
import os, sys
from globals import *
import requests
import shutil

class CullPDB():

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
        self.type = type

        #if dataset not 5926 or 6133 then set to 5926 default.

        #ensuring dataset is of type 6133 or 5926
        assert (self.type == 6133 or self.type ==5926), 'training datatset must be of type 6133 or 5926'

        #set training path and url class variables
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
        if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.train_path))):
            self.download_cullpdb()

        #load in dataset
        self.load_cullpdb()

    def load_cullpdb(self):

        print("Loading CullPDB {} training dataset (filtered: {})...\n".format(self.type, self.filtered))

        #load dataset
        data = np.load(os.path.join(os.getcwd(), DATA_DIR, self.train_path[:-3]))

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

                #one-hot encoding of val labels
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

                #one-hot encoding of test labels
                test_hot = np.ones((testhot.shape[0], testhot.shape[1])) #target array
                self.test_hot = get_onehot(testhot, test_hot)

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

                testhot = datahot[seq_index[5435:5690]]
                testlabel = labels[seq_index[5435:5690]]
                testpssm = datapssm[seq_index[5435:5690]]

                #one-hot encoding of test labels
                test_hot = np.ones((testhot.shape[0], testhot.shape[1])) #target array
                self.test_hot = get_onehot(testhot, test_hot)  #test_hot (target) not crrated


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
            self.testlabel = testlabel
            self.testpssm = testpssm

    def download_cullpdb(self):
        """
        Download Cullpdb dataset from Princeton URL and store locally in data directory

        Parameters
        ----------

        """
        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.train_path[:-3]))):

                #get data from url
                r = requests.get(self.url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()    #if response status code != 200

                dir_path = (os.path.join(os.getcwd(), DATA_DIR))
                source_path = (os.path.join(dir_path, self.train_path))
                destination_path = os.path.join(dir_path, self.train_path[:-3])

                #open local file
                open(source_path, 'wb').write(r.content)

                #export zipped dataset and store .npy file
                print('Exporting CullPDB {} Filtered = {} datatset....'.format(self.type, self.filtered))
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                #remove unrequired zipped version of dataset
                os.remove(source_path)

                print('Dataset downloaded successfully - stored in {} of size \
                    {} '.format(destination_path,self.dataset_size()))

            else:
                #dataset already present
                pass

        except OSError:
            print('Error downloading and exporting training dataset.\n')

    def __len__(self):
        """ Get number of proteins in CullPDB dataset - length of the 1st dimension """
        return (self.train_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CullPDB object """
        return ('CullPDB {} Training datatset - filtered: {}. \
            Shape of dataset: {}'.format(self.type,self.filtered, self.shape()))

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
        return str(round((os.path.getsize(os.path.join(os.getcwd(), DATA_DIR, self.train_path)))/(1024*1024))) + ' MB'

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
        if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path[:-3]))):
            self.download_cb513()

        #load test dataset
        CB513 = np.load(os.path.join(os.getcwd(), DATA_DIR, self.test_path[:-3]))

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

        #convert to one-hot encoding array
        test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
        test_hot = get_onehot(testhot, test_hot)

        #delete test data from ram
        del CB513

        #set cb513 class variables
        self.test_hot = test_hot
        self.testpssm = testpssm
        self.testlabel = testlabel

    def download_cb513(self):
        """ Download CASP11 dataset from Princeton URL and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path))):

                #get data from url
                r = requests.get(self.url, allow_redirects = True)
                r.raise_for_status()    #if response status code != 200

                dir_path = (os.path.join(os.getcwd(), DATA_DIR))
                source_path = (os.path.join(dir_path, self.test_path))
                destination_path = os.path.join(dir_path, self.test_path[:-3])

                #open local file
                open(source_path, 'wb').write(r.content)

                #export zipped dataset and store .npy file
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                #remove unrequired zipped version of dataset
                os.remove(source_path)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(destination_path,self.dataset_size()))

            else:
                #dataset already present
                pass

        except OSError:
            print('Error downloading and exporting dataset...\n')

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
        return str(round((os.path.getsize(os.path.join(os.getcwd(), DATA_DIR, self.test_path)))/(1024*1024))) + ' MB'

class CASP10():

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
        if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path))):
            self.download_casp10()

        #load casp10 dataset
        casp10_data = h5py.File((os.path.join(os.getcwd(), DATA_DIR, self.test_path)),'r')

        #load protein sequence and profile feature data
        casp10_data_hot = casp10_data['features'][:, :, 0:21]
        casp10_data_pssm = casp10_data['features'][:, :, 21:42]

        #load protein label data
        test_labels = casp10_data['labels'][:, :, 0:8]

        #crete one hot encoding vector for casp10 labels
        casp10_data_test_hot = np.ones((casp10_data_hot.shape[0], casp10_data_hot.shape[1]))
        casp10_data_test_hot = get_onehot(casp10_data_hot, casp10_data_test_hot)

        #delete test data from ram
        del casp10_data

        #set casp10 class variables
        self.casp10_data_test_hot = casp10_data_test_hot
        self.casp10_data_pssm = casp10_data_pssm
        self.test_labels = test_labels

    def download_casp10(self):
        """ Download CASP10 dataset from repository and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path))):

                #get data from url
                r = requests.get(self.url, allow_redirects = True)
                r.raise_for_status() #if response status code != 200

                dir_path = (os.path.join(os.getcwd(), DATA_DIR))
                source_path = (os.path.join(dir_path, self.test_path))

                #open local file
                open(source_path, 'wb').write(r.content)

                print('Dataset downloaded successfully - stored in {} of size \
                    {} '.format(source_path,self.dataset_size()))

            else:
                #dataset already present
                pass

        except OSError:
            print('Error downloading and exporting dataset...\n')

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
        return str(round((os.path.getsize(os.path.join(os.getcwd(), DATA_DIR,
            self.test_path)))/(1024*1024))) + ' MB'


class CASP11():

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
        if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path))):
            self.download_casp11()

        #load casp11 dataset
        casp11_data = h5py.File((os.path.join(os.getcwd(), DATA_DIR, self.test_path)),'r')

        #load protein sequence feature data
        casp11_data_hot = casp11_data['features'][:,:,0:21]
        # load profile feature data
        casp11_data_pssm = casp11_data['features'][:,:,21:42]
        #load protein label feature data
        test_labels = casp11_data['labels'][:,:,0:8]

        #convert to one-hot encoding array
        casp11_data_test_hot = np.ones((casp11_data_hot.shape[0], casp11_data_hot.shape[1]))
        casp11_data_test_hot = get_onehot(casp11_data_hot, casp11_data_test_hot)

        #delete test data from ram
        del casp11_data

        #set casp11 class variables
        self.casp11_data_test_hot = casp11_data_test_hot
        self.casp11_data_pssm = casp11_data_pssm
        self.test_labels = test_labels

    def download_casp11(self):
        """ Download CASP11 dataset from repository and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), DATA_DIR, self.test_path))):

                #get data from url
                r = requests.get(self.url, allow_redirects = True)
                r.raise_for_status()

                dir_path = (os.path.join(os.getcwd(), DATA_DIR))
                source_path = (os.path.join(dir_path, self.test_path))

                #open local file
                open(source_path, 'wb').write(r.content)

                print('Dataset downloaded successfully - stored in {} of size {} '.format(source_path,self.dataset_size()))

            else:
                #dataset already present
                pass

        except OSError:
            print('Error downloading and exporting dataset...\n')


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
        return str(round((os.path.getsize(os.path.join(os.getcwd(), DATA_DIR, self.test_path)))/(1024*1024))) + ' MB'

def get_onehot(source_array, target_array):
    """
    Description:
        Convert protein structure labels into one-hot encoding vector

    Parameters
    ----------
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
