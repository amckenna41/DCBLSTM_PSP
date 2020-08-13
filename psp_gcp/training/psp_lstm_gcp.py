#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="practical-6-259701-4d5aef3b333e.json"
# os.environ["BUCKET_NAME"]="keras-python-models"
# os.environ["REGION"]="us-central1"
# os.environ["JOB_NAME"] = "my_first_keras_job"
# os.environ["JOB_DIR"] = "gs/keras-python-models/"

# # Please note that Python 2.7 which is used in Tensorflow works with relative imports.
# It is possible that you locally used Python 3.4 which worked with absolute imports. T
# hat is why it worked locally but not on Google Cloud. You can refer to this post to
# modify your import statement. So, if you include the line “from __future__ import
#  absolute_import” at the top of your code, before the line “import tensorflow as tf” ,
#  your code may work.
import numpy as np
import gzip
import h5py
import tensorflow as tf
from tensorflow import keras
import argparse
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, Concatenate, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import random as rn
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io
import os
import sys
import importlib
#import os.path as path
#
#
# sys.path.append('../../')
# from models.psp_lstm_model import *
# from models.psp_gru_model import *
# from data.load_dataset import *
# from data.get_dataset import *

print('cwd from top of psp_lstm_gcp file ', os.getcwd())
# sys.path.append('../')
# sys.path.append('/Users/adammckenna/protein_structure_prediction_DeepLearning/data')
# sys.path.append('/Users/adammckenna/protein_structure_prediction_DeepLearning/models')
#
# from .. models.psp_lstm_model import *
# from .. models.psp_gru_model import *
# from .. data.load_dataset import *
# from .. data.get_dataset import *
# spec = importlib.util.spec_from_file_location("module.", "/path/to/file.py")
# foo = importlib.util.module_from_spec(spec)
# # spec.loader.exec_module(foo)
# from protein_structure_prediction_DeepLearning.models import psp_lstm_model, psp_gru_model
# from protein_structure_prediction_DeepLearning.data import get_dataset, load_dataset
# from google.cloud import storage
# from protein_structure_prediction_DeepLearning.models import psp_lstm_model
# from protein_structure_prediction_DeepLearning.data import load_dataset, get_dataset
# data_dir = path.abspath(path.join(__file__ ,"../../../data"))
# sys.path.append(data_dir )
# from load_dataset import *
#
# model_dir = path.abspath(path.join(__file__ ,"../../../models"))
# sys.path.append(model_dir)
# from psp_lstm_model import *

BUCKET_NAME = "keras-python-models"
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'

#Data and models are packages due to the __init__ - they contain modules, e.g psp_lstm_model.py, get_dataset.py
# client = storage.Client()
# bucket = client.get_bucket(BUCKET_NAME)

#load filtered cullpdb training data
def load_cul6133_filted_2():

    # storage_client = storage.Client.from_service_account_json('psp-keras-training.json')
    print("Loading training dataset (Cullpdb_filtered)...\n")

    # #download dataset if not already in current directory
    # if not (os.path.isfile(TRAIN_PATH)):
    #     #if training data not present then download to current working dir
    #     get_dataset.get_cullpdb_filtered()

    # cur_path = os.path.dirname(__file__)
    # new_path = os.path.relpath('../data/TRAIN_PATH', cur_path)
    #load dataset
    # data = np.load('gs://keras-python-models/'+TRAIN_PATH)
    #
    # bucket = storage_client.bucket(BUCKET_NAME)
    # g = bucket.get_blob(TRAIN_PATH)
    # g_download = g.download_as_string()
    f = BytesIO(file_io.read_file_to_string('gs://keras-python-models/cullpdb+profile_6133_filtered.npy', binary_mode=True))
    # f = BytesIO(file_io.read_file_to_string(g_download, binary_mode=True))

    data = np.load(f)

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
    trainhot = datahot[seq_index[:1320]]
    trainlabel = labels[seq_index[:1320]]
    trainpssm = datapssm[seq_index[:1320]]

    #get validation data
    vallabel = labels[seq_index[1320:1384]] #8
    valpssm = datapssm[seq_index[1320:1384]] # 21
    valhot = datahot[seq_index[1320:1384]] #21

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

# def main(job_dir, **args):
# def main(job_dir, args):
def main():

    print('cwd from psp_lstm_gcp dir ', os.getcwd())

    job_dir = os.environ["JOB_DIR"]

    logs_path = job_dir + 'logs/tensorboard'

    with tf.device('/device:GPU:0'):
        #Load data
        print(os.getcwd())
        #train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted_2()

        #build model
        #model = build_model()
        #model = build_model_lstm()
        model = load_model('model_1')
        #do google cloud authentication for outside users calling function
        print(os.getcwd())
        #
        # batch_size = str(args.batch_size)
        # epochs = str(args.epochs)
        # job_dir = str(args.job_dir)
        # storage_bucket = str(args.storage_bucket)
        # print(batch_size, epochs)
        #subprocess.call("chmod +x gcp_deploy.sh", shell=True)
        #create storage bucket if doesn't exist
        # os.environ["BATCH_SIZE"] = batch_size
        # os.environ["EPOCHS"]= epochs
        # os.environ["JOB_DIR"] = job_dir
        # os.environ["STORAGE_BUCKET"] = storage_bucket
            # call(['bash', 'run.sh', batch_size, epochs])
            # process = subprocess.run('./gcp_deploy.sh',  check=True, timeout=10, env ={"BATCH_SIZE": batch_size, "EPOCHS":epochs})
            #chmod +x gcp_deploy.sh
        #subprocess.call(["../gcp_deploy.sh"],shell =True)
            # subprocess.Popen("./gcp_deploy.sh", shell =True, env ={"BATCH_SIZE": batch_size, "EPOCHS":epochs})
        print(os.getcwd())
        #add callbacks for tesnorboard and history
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        print(os.environ.get('BATCH_SIZE'))
        print(os.environ.get('EPOCHS'))
        batch_size = os.environ.get('BATCH_SIZE')
        epochs = os.environ.get('EPOCHS')

        #fit model
        print('Fitting model...')
        model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],shuffle=True)

        #save model
        print('Saving model')
        model.save('model_1.h5')
        with file_io.FileIO('model_1.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model_1.h5', mode='w+') as output_f:
                output_f.write(input_f.read())

main()
# ##Running the app
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
#     parser.add_argument('-b', '--batch_size', type=int, default=42,
#                         help='batch size for training data (default: 42)')
#     # parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
#     #                     help='input batch size for testing (default: 1024)')
#     parser.add_argument('--data_dir', type=str, default='../data',
#                         help='Directory for training and test datasets')
#     parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
#                         help='Google Storage Bucket storing data and logs')
#     # parser.add_argument('--result_dir', type=str, default='./result',
#     #                     help='Output directory (default: ./result)')
#     # parser.add_argument('--seed', type=int, default=1, metavar='S',
#     #                     help='random seed (default: 1)')
#     parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
#                         help ='The number of nodes for first LSTM hidden layer')
#     parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
#                         help ='The number of nodes for second LSTM hidden layer')
#     parser.add_argument('-dr', '--dropout', type=float, default = 0.5,
#                         help='The dropout applied to input (default = 0.5)')
#     parser.add_argument('-op', '--optimizer', default = 'adam',
#                         help='The optimizer used in compiling and fitting the models')
#     parser.add_argument('-e', '--epochs', type=int, default=10,
#                         help='The number of epochs to run on the model')
#     parser.add_argument('-jd', '--job_dir', help='GCS location to write checkpoints and export models',required=False,
#                         default = 'gs://keras-python-models')
#     args = parser.parse_args()
#     # parser = argparse.ArgumentParser()
#     #
#     # # Input Arguments
#     # parser.add_argument(
#     #   '--job-dir',
#     #   help='GCS location to write checkpoints and export models',
#     #   required=True
#     # )
#     # args = parser.parse_args()
#     # arguments = args.__dict__
#     #
#
#     main(args.job_dir, args)
    # main(**arguments)
#gcp shell script calls the psp_lstm_gcp script, does not have to be a main function ??
