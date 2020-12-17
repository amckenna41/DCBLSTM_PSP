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

class CulPDB6133(object):

    """
        Description:
            CulPDB6133 class creates instance of the training dataset.

        Args:
            all_data (float): The proportion of the training data to use, must be in range 0.5 to 1.0, default: 1.0
            filtered (bool): What PDB training dataset to use, filtered or unfiltered, default = True.

        Returns:
            CulPDB6133 training dataset object
    """

    def __init__(self, all_data =1, filtered = True):

        self.all_data = all_data
        self.filtered = filtered

        print("Loading training dataset (Cullpdb_filtered)...\n")

        if (self.filtered):
            self.train_path = TRAIN_PATH_FILTERED_NPY
        else:
            self.train_path = TRAIN_PATH_UNFILTERED_NPY

        #download dataset if not already in current directory
        # if not (os.path.isfile(os.getcwd() + '/data/' + self.train_path)):
        if not (os.path.isfile(os.path.join(os.getcwd(), 'data', self.train_path))):
            # get_cullpdb_filtered()
            print("getting dataset now...")
            self.get_cullpdb(self.filtered)

        #load dataset
        # data = np.load(self.train_path)
        data = np.load(os.path.join(os.getcwd(), 'data', self.train_path))

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

        #calculate the indexes for each dimension based on all_data input parameter
        data_index = int(5278 * all_data)
        val_data_index =  int(256 * all_data)
        val_data_upper = data_index + val_data_index

        #get training data
        trainhot = datahot[seq_index[:data_index]]
        trainlabel = labels[seq_index[:data_index]]
        trainpssm = datapssm[seq_index[:data_index]]

        #get validation data
        vallabel = labels[seq_index[data_index:val_data_upper]] #8
        valpssm = datapssm[seq_index[data_index:val_data_upper]] # 21
        valhot = datahot[seq_index[data_index:val_data_upper]] #21


        train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
        for i in range(trainhot.shape[0]):
            for j in range(trainhot.shape[1]):
                if np.sum(trainhot[i,j,:]) != 0:
                    train_hot[i,j] = np.argmax(trainhot[i,j,:])


        val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
        for i in range(valhot.shape[0]):
            for j in range(valhot.shape[1]):
                if np.sum(valhot[i,j,:]) != 0:
                    val_hot[i,j] = np.argmax(valhot[i,j,:])

        #delete training data from ram
        del data

        self.train_hot = train_hot
        self.trainpssm = trainpssm
        self.trainlabel = trainlabel
        self. val_hot = val_hot
        self.valpssm = valpssm
        self.vallabel = vallabel

    def get_cullpdb(self, filtered):
        """ Download Cullpdb dataset from Princeton URL and store locally in data directory """

        if (filtered):
            train_path = TRAIN_PATH_FILTERED
            train_url = TRAIN_FILTERED_URL
        else:
            train_path = TRAIN_PATH_UNFILTERED
            train_url = TRAIN_UNFILTERED_URL

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), 'data', train_path))):

                r = requests.get(train_url, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = (os.path.join(os.getcwd(), 'data'))
                source_path = (os.path.join(dir_path, train_path))
                destination_path = os.path.join(dir_path, train_path[:-3])

                open(source_path, 'wb').write(r.content)

                print('Exporting Cullpdb 6133 datatset....')
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                os.remove(os.path.join(os.getcwd(), 'data', train_path))

            else:
                #dataset already present
                print(str(self) + " already present...")

        except OSError:
            print('Error downloading and exporting training dataset\n')

    #get length of CullPDB training dataset
    def __len__(self):
        """ Get number of proteins in CullPDB dataset - length of the 1st dimension """
        return (self.train_hot[:,0].shape[0])

    def __str__(self):
        """ Print string representation of CullPDB object """
        return ('CullPDB6133 Training datatset - filtered: {}'.format(self.filtered))

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

        print("Loading test dataset (CB513)...\n")

        #download dataset if not already in current directory
        # if not (os.path.isfile(TEST_PATH)):
        if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CB513_NPY))):
            self.get_cb513()

        #load test dataset
        CB513 = np.load(os.path.join(os.getcwd(), 'data', CB513_NPY))

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
        for i in range(testhot.shape[0]):
            for j in range(testhot.shape[1]):
                if np.sum(testhot[i,j,:]) != 0:
                    test_hot[i,j] = np.argmax(testhot[i,j,:])

        #delete test data from ram
        del CB513

        self.test_hot = test_hot
        self.testpssm = testpssm
        self.testlabel = testlabel

    def get_cb513(self):
        """ Download CASP11 dataset from Princeton URL and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CB513_PATH))):

                r = requests.get(CB513_URL, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = (os.path.join(os.getcwd(), 'data'))
                source_path = (os.path.join(dir_path, CB513_PATH))
                destination_path = os.path.join(dir_path, CB513_PATH[:-3])

                open(source_path, 'wb').write(r.content)

                print('Exporting CB513 datatset....')
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)

                os.remove(os.path.join(os.getcwd(), 'data', CB513_PATH))

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
        return ('CB513 Test datatset')

    def shape(self):
        """ Output shape of CB513 object """
        return self.test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CB513 dataset specified by protein index argument """
        labels = self.testlabel[protein_index,:,:]
        return labels


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

        print("Loading CASP10 dataset...\n")

        #download dataset if not already in current directory
        # if not (os.path.isfile(CASP10_PATH)):
        if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CASP10_PATH))):
            self.get_casp10()

        #load casp10 dataset
        # casp10_data = h5py.File(CASP10_PATH, 'r')
        casp10_data = h5py.File((os.path.join(os.getcwd(), 'data', CASP10_PATH)),'r')

        #load protein sequence and profile feature data
        casp10_data_hot = casp10_data['features'][:, :, 0:21]
        casp10_data_pssm = casp10_data['features'][:, :, 21:42]

        #load protein label data
        test_labels = casp10_data['labels'][:, :, 0:8]

        #create new protein sequence feature, set values to max value if if value!=0 ?
        casp10_data_test_hot = np.ones((casp10_data_hot.shape[0], casp10_data_hot.shape[1]))
        for x in range(casp10_data_hot.shape[0]):
            for y in range(casp10_data_hot.shape[1]):
                   if np.sum(casp10_data_hot[x,y,:]) != 0:
                        casp10_data_test_hot[x,y] = np.argmax(casp10_data_hot[x,y,:])

        print('CASP10 dataset loaded...\n')

        #delete test data from ram
        del casp10_data

        self.casp10_data_test_hot = casp10_data_test_hot
        self.casp10_data_pssm = casp10_data_pssm
        self.test_labels = test_labels


    def get_casp10(self):
        """ Download CASP10 dataset from repository and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CASP10_PATH))):

                r = requests.get(CASP10_URL, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = (os.path.join(os.getcwd(), 'data'))
                source_path = (os.path.join(dir_path, CASP10_PATH))
                destination_path = os.path.join(dir_path, CASP10_PATH[:-3])

                open(source_path, 'wb').write(r.content)

                print('Exporting CASP10 datatset....')
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)
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
        return ('CASP10 Test datatset')

    def shape(self):
        """ Output shape of CASP10 object """
        return self.casp10_data_test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP10 dataset specified by protein index argument """
        labels = self.test_labels[protein_index,:,:]
        return labels



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

        print("Loading CASP11 dataset...\n")


        #download dataset if not already in current directory
        # if not (os.path.isfile(CASP11_PATH)):
        if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CASP11_PATH))):
            self.get_casp11()

        #load casp11 dataset
        # casp11_data = h5py.File(CASP11_PATH, 'r')
        casp11_data = h5py.File((os.path.join(os.getcwd(), 'data', CASP11_PATH)),'r')

        #load protein sequence and profile feature data
        casp11_data_hot = casp11_data['features'][:,:,0:21]
        casp11_data_pssm = casp11_data['features'][:,:,21:42]
        #load protein label data
        test_labels = casp11_data['labels'][:,:,0:8]

        #create new protein sequence feature, set values to max value if if value!=0 ?
        casp11_data_test_hot = np.ones((casp11_data_hot.shape[0], casp11_data_hot.shape[1]))
        for x in range(casp11_data_hot.shape[0]):
            for y in range(casp11_data_hot.shape[1]):
                if np.sum(casp11_data_hot[x,y,:]) != 0:
                    casp11_data_test_hot[x,y] = np.argmax(casp11_data_hot[x,y,:])

        print('CASP11 dataset loaded...\n')

        #delete test data from ram
        del casp11_data

        self.casp11_data_test_hot = casp11_data_test_hot
        self.casp11_data_pssm = casp11_data_pssm
        self.test_labels = test_labels

    def get_casp11(self):
        """ Download CASP11 dataset from repository and store locally in data directory """

        try:
            if not (os.path.isfile(os.path.join(os.getcwd(), 'data', CASP11_PATH))):

                r = requests.get(CASP11_URL, allow_redirects = True) #error handling, if response == 200
                r.raise_for_status()

                dir_path = (os.path.join(os.getcwd(), 'data'))
                source_path = (os.path.join(dir_path, CASP11_PATH))
                destination_path = os.path.join(dir_path, CASP11_PATH[:-3])

                open(source_path, 'wb').write(r.content)

                print('Exporting CASP11 datatset....')
                with gzip.open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfile(source_path, destination_path)
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
        return ('CASP11 Test datatset')

    def shape(self):
        """ Output shape of CASP11 object """
        return self.casp11_data_test_hot.shape

    def get_data_labels(self, protein_index):
        """ Get data labels in CASP11 dataset specified by protein index argument """
        labels = self.test_labels[protein_index,:,:]
        return labels

#download all datasets used in PSP
def download_all_data():


    """
        Description:
            Create instance for each training and test datasets used in project. Calling this function can
            download all required datasets and store in data directory.
        Args:
            None
        Returns:
            None
    """

    cul6133 = CulPDB6133()
    cb513 = CB513()
    casp10 = CASP10()
    casp11 = CASP11()
