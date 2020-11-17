#Loading training and test datasets

#importing libraries and dependancies
import numpy as np
import gzip
import h5py
import os
try:
    from get_dataset import *
except:
    from data.get_dataset import *


#File paths for train and test datasets in data dir
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'
TEST_PATH = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'


#load filtered cullpdb training data, all_data - proportion of data to load in, default is 1 which is whole dataset
def load_cul6133_filted(all_data=1):

    print("Loading training dataset (Cullpdb_filtered)...\n")

    #download dataset if not already in current directory
    if not (os.path.isfile(TRAIN_PATH)):
        get_cullpdb_filtered()

    #load dataset
    data = np.load(TRAIN_PATH)
    #reshape dataset
    data = np.reshape(data, (-1, 700, 57))
    #sequence feature
    datahot = data[:, :, 0:21]
    #profile features
    datapssm = data[:, :, 35:56]
    #secondary struture labels
    labels = data[:, :, 22:30]
    #np.random.seed(2018)

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

    return train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel


#loading CB513 test dataset
def load_cb513(all_data = 1):

    print("Loading test dataset (CB513)...\n")

    #download dataset if not already in current directory
    if not (os.path.isfile(TEST_PATH)):
        get_cb513()

    #load test dataset
    CB513= np.load(TEST_PATH)
    #reshape dataset
    CB513= np.reshape(CB513,(-1,700,57))
    #sequence feature
    testhot=CB513[:, :, 0:21]
    #profile feature
    testpssm=CB513[:, :, 35:56]
    #secondary struture label
    testlabel = CB513[:, :, 22:30]

    #calculate the indexes for each dimension based on all_data input parameter
    test_data_index = int(514 * all_data)

    testhot = testhot[:test_data_index]
    testpssm = testpssm[:test_data_index]
    testlabel = testlabel[:test_data_index]

    #convert to one-hot array
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])

    #delete test data from ram
    del CB513
    return test_hot, testpssm, testlabel

#load CASP10 test dataset from cwd
def load_casp10():

    print("Loading CASP10 dataset...\n")

    #download dataset if not already in current directory
    if not (os.path.isfile(CASP10_PATH)):
        get_casp10()

    #load casp10 dataset
    casp10_data = h5py.File(CASP10_PATH, 'r')

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

    return casp10_data_test_hot, casp10_data_pssm, test_labels

#load CASP11 test dataset from cwd
def load_casp11():

    print("Loading CASP11 dataset...\n")

    #download dataset if not already in current directory
    if not (os.path.isfile(CASP11_PATH)):
        get_casp11()

    #load casp11 dataset
    casp11_data = h5py.File(CASP11_PATH, 'r')

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

    return casp11_data_test_hot, casp11_data_pssm, test_labels

#download all datasets used in PSP
def download_all_data():

    load_cul6133_filted()
    load_cb513()
    load_casp10()
    load_casp11()

#if script called by itself then all datasets are downloaded and stored in /data dir
if __name__ == '__main__':

    print('Downloading all datasets required in this project...')
    # download_all_data()
