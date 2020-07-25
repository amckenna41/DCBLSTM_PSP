#Loading training and test datasets

#importing libraries and dependancies
import numpy as np
import gzip
import h5py
import os
from get_dataset import *

#File paths for train and test datasets
TRAIN_PATH = '/cullpdb+profile_6133_filtered.npy'
TEST_PATH = '/cb513+profile_split1.npy'
CASP10_PATH = '/casp10.h5'
CASP11_PATH = '/casp11.h5'



def load_cul6133_filted():

    print("Loading training dataset (Cullpdb_filtered)...")
    if not (os.path.isfile(TRAIN_PATH)):
        #if training data not present then download to current working dir
        get_dataset.download_export_dataset()

    #load dataset
    data = np.load('cullpdb+profile_6133_filtered.npy')

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

    #get training data
    trainhot = datahot[seq_index[:5278]]
    trainlabel = labels[seq_index[:5278]]
    trainpssm = datapssm[seq_index[:5278]]

    #get validation data
    vallabel = labels[seq_index[5278:5534]] #8
    valpssm = datapssm[seq_index[5278:5534]] # 21
    valhot = datahot[seq_index[5278:5534]] #21

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

    return train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel


def load_cb513():

    print("Loading test dataset (CB513)...")
    if not (os.path.isfile(TRAIN_PATH)):
        get_dataset.download_export_dataset()

    CB513= np.load('cb513+profile_split1_2.npy')
    CB513= np.reshape(CB513,(-1,700,57))

    datahot=CB513[:, :, 0:21]#sequence feature
    datapssm=CB513[:, :, 35:56]#profile feature

    labels = CB513[:, :, 22:30] # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm

    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])

    return test_hot, testpssm, testlabel
    pass


def load_casp10():
    def load_casp10_data():

    #load casp10 dataset
    casp10_data = h5py.File("casp10.h5")

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

    return casp10_data_test_hot, casp10_data_pssm, test_labels

    pass


def load_casp11():
    pass
