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
import training.training_utils.gcp_utils as utils

#File names for train and test datasets
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'cb513+profile_split1.npy.gz'
TRAIN_NPY = 'cullpdb+profile_6133_filtered.npy'
TEST_NPY = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'

#URL's for train and test dataset download
TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"
CASP_10_URL = "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5"
CASP_11_URL = "https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5"
BUCKET_NAME = "gs://keras-python-models-2"

#download and unzip filtered cullpdb training data
def get_cullpdb_filtered():

    print('Downloading Cullpdb 6133 dataset...\n')

    try:
        print('CWD in get_cullpdb_filtered - {}'.format(os.getcwd()))
        if not (os.path.isfile(os.getcwd() + '/' + TRAIN_PATH)):
            r = requests.get(TRAIN_URL, allow_redirects = True)
            open(TRAIN_PATH, 'wb').write(r.content)
            dir_path = os.path.dirname(os.path.realpath(TRAIN_PATH))
            source_path = dir_path + '/' + TRAIN_PATH
            destination_path = dir_path + '/' + TRAIN_NPY

            print('Exporting Cullpdb 6133 datatset....')
            with gzip.open(TRAIN_PATH, 'rb') as f_in:
                with open(TRAIN_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('Cullpdb 6133 dataset already present...\n')

    except OSError:
        print('Error downloading and exporting training dataset\n')
        return

    #uploading training data to GCP Storage
    blob_path = BUCKET_NAME + '/data/' + TRAIN_NPY
    utils.upload_file(blob_path, TRAIN_NPY)


#download and unzip CB513 test data
def get_cb513():

    print('Downloading CB513 dataset...\n')

    try:
        if not (os.path.isfile(os.getcwd() + '/' + TEST_PATH)):
            r = requests.get(TEST_URL, allow_redirects = True)
            open(TEST_PATH, 'wb').write(r.content)
            dir_path = os.path.dirname(os.path.realpath(TEST_PATH))
            source_path = dir_path + '/' + TEST_PATH
            destination_path = dir_path + '/' + TEST_NPY

            print('Exporting CB513 datatset...\n')
            with gzip.open(TEST_PATH, 'rb') as f_in:
                with open(TEST_NPY, 'wb') as f_out:
                    shutil.copyfile(source_path, destination_path)
        else:
            print('CB513 dataset already present...\n')

    except OSError:
        print('Error downloading and exporting testing dataset\n')
        return

    #uploading test data to GCP Storage
    blob_path = BUCKET_NAME + '/data/' + TEST_NPY
    utils.upload_file(blob_path, TEST_NPY)

#downloading CASP10 test dataset
def get_casp10():

    try:
        if not (os.path.isfile(CASP10_PATH)):
            #os.system('wget -O {} {}'.format(CASP10_PATH, CASP_10_URL))
            r = requests.get(CASP_10_URL, allow_redirects = True)
            open('casp10.h5', 'wb').write(r.content)
            print('CASP10 dataset downloaded\n')

        else:
            print('CASP10 dataset already exists\n')

    except OSError:
        print('Error downloading and exporting CASP10 dataset\n')

    #uploading test data to GCP Storage
    blob_path = BUCKET_NAME + '/data/' + CASP10_PATH
    utils.upload_file(blob_path, CASP10_PATH)

#downloading CASP11 test dataset
def get_casp11():

    try:
        if not (os.path.isfile(CASP11_PATH)):
            r = requests.get(CASP_11_URL, allow_redirects = True)
            open('casp11.h5', 'wb').write(r.content)
            print('CASP11 dataset downloaded\n')

        else:
            print('CASP11 dataset already exists\n')

    except OSError:
        print('Error downloading and exporting CASP11 dataset\n')

    #uploading test data to GCP Storage
    blob_path = BUCKET_NAME + '/data/' + CASP11_PATH
    utils.upload_file(blob_path, CASP11_PATH)

#load filtered cullpdb training data
def load_cul6133_filted(all_data = 1):

    print("Loading training dataset (Cullpdb_filtered)...\n")

    TRAIN_PATH_ = os.getcwd() + '/' + TRAIN_PATH

    #if dataset not in directory then call get function
    if not (os.path.isfile(TRAIN_PATH_)):
        print('Getting dataset')
        get_cullpdb_filtered()

    data = np.load(TRAIN_PATH_)

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
def load_cb513(all_data = 1):

    print("Loading test dataset (CB513)...\n")

    TEST_PATH_ = os.getcwd() + '/' + TEST_PATH

    #if dataset not in directory then call get function
    if not (os.path.isfile(TEST_PATH_)):
        print('Getting dataset')
        get_cb513()

    #load test dataset
    CB513 = np.load(TEST_PATH_)

    #reshape dataset
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

    CASP10_PATH_ = os.getcwd() + '/' + CASP10_PATH

    #if dataset not in directory then call get function
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

    CASP11_PATH_ = os.getcwd() + '/' + CASP11_PATH

    #if dataset not in directory then call get function
    if not (os.path.isfile(CASP11_PATH_)):
        get_casp11()

    #load casp11 dataset
    casp11_data = h5py.File(CASP11_PATH_)

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

# #download and load all datasets used in PSP
# def download_all_data():
#
#     load_cul6133_filted()
#     load_cb513()
#     load_casp10()
#     load_casp11()
#
# if __name__ == "main":
#     download_all_data()
