#Loading training and test datasets

#importing libraries and dependancies
import numpy as np
import gzip
import h5py
import os
from data.get_dataset import *

#File paths for train and test datasets
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'
TEST_PATH = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'


#load filtered cullpdb training data
def load_cul6133_filted(all_data=1):

    print("Loading training dataset (Cullpdb_filtered)...\n")

    #Below code allows for function to be called from data dir or top-level dir
    cwd = os.getcwd()
    print(cwd)
    # if cwd[len(cwd)-4:len(cwd)] != 'data':
    #     os.chdir('data')
    #     new_cwd = os.getcwd() #now in the data dir
    #     #TRAIN_PATH_ = new_cwd + '/' + TRAIN_PATH

    #validation if load_dataset called from models dir
    # os.chdir('..')
    os.chdir('data') #change cwd to data dir 
    new_cwd = os.getcwd() #now in the data dir
    TRAIN_PATH_ = new_cwd + '/' + TRAIN_PATH
    TRAIN_PATH_ = os.getcwd() + '/' + TRAIN_PATH

    if not (os.path.isfile(TRAIN_PATH_)):
        print('Getting dataset')
        get_cullpdb_filtered()

    data = np.load(TRAIN_PATH_)


    #change back direcotry to main
    # #download dataset if not already in current directory
    # if not (os.path.isfile(TRAIN_PATH)):
    #     #if training data not present then download to current working dir
    #     get_dataset.get_cullpdb_filtered()

    # cur_path = os.path.dirname(__file__)
    # new_path = os.path.relpath('../data/TRAIN_PATH', cur_path)
    # d = os.getcwd()
    # print(d)
    # os.chdir("..")
    # os.chdir("..")
    # print(os.getcwd())
    # o = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))] # Gets all directories in the folder as a tuple
    # for item in o:
    #     if os.path.exists(item + 'TRAIN_PATH'):
    #         file = item + 'TRAIN_PATH'
    #         print(file)
    # d = os.getcwd()
    # if d[len(d)-4:len(d)] != 'data':
    #     os.chdir('data')
    #     new_d = os.getcwd()
    #     TRAIN_PATH_ = new_d + '/' + TRAIN_PATH
    #     print(TRAIN_PATH_)
    #     print('CWD Pre Load - {}\n'.format(os.getcwd()))
    #     data = np.load(TRAIN_PATH_)
    # else:
    #     print(TRAIN_PATH)
    #     print('CWD Pre Load - {}'.format(os.getcwd()))
    #     data = np.load(TRAIN_PATH)
    #load dataset
    #data = np.load(TRAIN_PATH)

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


    return train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel

#loading CB513 test dataset
def load_cb513(all_data=1):

    print("Loading test dataset (CB513)...\n")

    #Below code allows for function to be called from data dir or top-level dir
    cwd = os.getcwd()
    if cwd[len(cwd)-4:len(cwd)] != 'data':
        os.chdir('data')
        new_cwd = os.getcwd() #now in the data dir
        #TRAIN_PATH_ = new_cwd + '/' + TRAIN_PATH

    TEST_PATH_ = os.getcwd() + '/' + TEST_PATH

    if not (os.path.isfile(TEST_PATH_)):
        print('Getting dataset')
        get_cb513()

    CB513 = np.load(TEST_PATH_)

    #download dataset if not already in current directory
    # if not (os.path.isfile(TEST_PATH)):
    #     get_dataset.get_cb513()

    #load test dataset
    #CB513= np.load(TEST_PATH)
    CB513= np.reshape(CB513,(-1,700,57))

    #sequence feature
    testhot=CB513[:, :, 0:21]
    #profile feature
    testpssm=CB513[:, :, 35:56]
    #secondary struture label
    testlabel = CB513[:, :, 22:30]

    test_data_index = int(514 * all_data)

    testhot = testhot[:test_data_index]
    testpssm = testpssm[:test_data_index]
    testlabel = testlabel[:test_data_index]

    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])

    return test_hot, testpssm, testlabel

#load CASP10 test dataset
def load_casp10():

    print("Loading CASP10 dataset...\n")

    #Below code allows for function to be called from data dir or top-level dir
    cwd = os.getcwd()
    if cwd[len(cwd)-4:len(cwd)] != 'data':
        os.chdir('data')
        new_cwd = os.getcwd() #now in the data dir
        #TRAIN_PATH_ = new_cwd + '/' + TRAIN_PATH

    CASP10_PATH_ = os.getcwd() + '/' + CASP10_PATH

    if not (os.path.isfile(CASP10_PATH_)):
        print('Getting dataset')
        get_casp10()

    #load casp10 dataset
    casp10_data = h5py.File(CASP10_PATH_)

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

    return casp10_data_test_hot, casp10_data_pssm, test_labels

#load CASP11 test dataset
def load_casp11():

    print("Loading CASP11 dataset...\n")

    #download dataset if not already in current directory
    if not (os.path.isfile(CASP11_PATH)):
        get_dataset.get_casp11()

    #load casp11 dataset
    casp11_data = h5py.File(CASP11_PATH)

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

    return casp11_data_test_hot, casp11_data_test_hot, test_labels

#download all datasets used in PSP
def download_all_data():

    load_cul6133_filted()
    load_cb513()
    load_casp10()
    load_casp11()

if __name__ == 'main':

    #Will this script really be called by itself??
    download_all_data()
