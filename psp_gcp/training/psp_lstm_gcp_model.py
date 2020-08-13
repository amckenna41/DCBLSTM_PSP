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
#from tensorflow import keras
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import random as rn
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io
import os
import sys
import importlib
from datetime import date
from datetime import datetime
from google.cloud import storage
import subprocess
import matplotlib.pyplot as plt
from training.get_dataset import *
print(os.getcwd())
d = os.getcwd()
g = d + '/' + 'training/psp_keras_training.json'
print(g)

# os.chdir('training')
# print('cd new dir', os.getcwd())
import json
from google.oauth2 import service_account

# fd, path = tempfile.mkstemp()
# with os.fdopen(fd, 'w') as tmp:
#     tmp.write(jsonfile)
# credentials = service_account.Credentials.from_service_account_file(path)
# storage_client = storage.Client(credentials=credentials)
# bucket = storage_client.get_bucket("keras-python-models")
# blob = bucket.blob('models/' + filename)
# for blob in blobs:
#     print(blob.name)

print('Doing something')
# creds = service_account.Credentials.from_service_account_file('/Users/adammckenna/protein_structure_prediction_DeepLearning/psp_gcp_test_dir/training/psp_keras_training.json')
# storage_client = storage.Client.from_service_account_json(
#         'psp_keras_training.json')
# storage_client = storage.Client.from_service_account_json(
#         'service-account.json')
# storage_client = storage.Client(project='psp-keras-training', credentials=creds)
# print(os.path.abspath('psp_keras_training.json'))
# print(os.getcwd())
# data = json.load(open('psp_keras_training.json'))
# storage_client = storage.Client.from_service_account_json("training/psp_keras_training.json", 'psp-keras-training')#./psp_keras_training.json, psp_keras_training.json, /training/psp_keras_training.json, ./training/psp_keras_training.json, training/psp_keras_training.json
# # client = storage.Client.from_service_account_json('/path/to/keyfile.json', 'project')
#
# bucket = storage_client.get_bucket("keras-python-models")
# #
# filename = "model_1.h5"
# blob = bucket.blob(filename)
# blob = bucket.blob('models/' + filename)
#import os.path as path
#
#
# sys.path.append('../../')
# from models.psp_lstm_model import *
# from models.psp_gru_model import *
# from data.load_dataset import *
# from data.get_dataset import *
from tensorflow.core.protobuf import rewriter_config_pb2
# from tensorflow.keras.backend import set_session
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.compat.v1.keras.backend import set_session

# config_proto = tf.ConfigProto()
config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
# session = tf.Session(config=config_proto)
session = tf.compat.v1.Session(config=config_proto)

set_session(session)

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

    # #get training data
    # trainhot = datahot[seq_index[:1320]]
    # trainlabel = labels[seq_index[:1320]]
    # trainpssm = datapssm[seq_index[:1320]]
    trainhot = datahot[seq_index[:5278]]
    trainlabel = labels[seq_index[:5278]]
    trainpssm = datapssm[seq_index[:5278]]

    vallabel = labels[seq_index[5278:5534]] #8
    valpssm = datapssm[seq_index[5278:5534]] # 21
    valhot = datahot[seq_index[5278:5534]]
    #get validation data
    # vallabel = labels[seq_index[1320:1384]] #8
    # valpssm = datapssm[seq_index[1320:1384]] # 21
    # valhot = datahot[seq_index[1320:1384]] #21

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

def build_model():

                #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(700,21), name='aux_input')
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)

        #get shape of input layers
    print (main_input.get_shape())
    print (auxiliary_input.get_shape())

    #concatenate input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    #1D Convolutional layer and maxpooling to downsample features
    conv1_features = Conv1D(42,1,strides=1,activation='relu', padding='same')(concat)
    max_pool_1d = MaxPooling1D(pool_size = 2, strides =1, padding='same')(conv1_features)
    conv1_features = Reshape((700, 42, 1))(max_pool_1d)

    # print ('conv1_features shape', conv1_features.get_shape())
    #print ('max_pool_1d shape', max_pool_1d.get_shape())

    #reshape 1D convolutional layer
    conv1_features = Reshape((700, 42, 1))(max_pool_1d)

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(conv1_features)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('MaxPooling Shape', max_pool_2D.get_shape())

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(84,3,strides=1,activation='relu', padding='same')(max_pool_2D)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    #conv2_batch_normal = BatchNormalization()(conv2_features)
    #max_pool_2D = MaxPooling2D(pool_size=(1, 2), strides=None, border_mode='same')(conv2_batch_normal)

    #conv2_features = Convolution2D(84,3,1,activation='relu', padding='same')(conv2_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()

    #reshape 2D convolutional layers
    conv2_features = Reshape((700, 84*42))(max_pool_2D)

    #conv2_features = Dense(500, activation='relu')(conv2_features)

    #Long Short Term Memory layers with tanh activation, sigmoid recurrent activiation and dropout of 0.5
    lstm_f1 = LSTM(400,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)
    lstm_b1 = LSTM(400, return_sequences=True, activation='tanh',go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)

    lstm_f2 = LSTM(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_f1)
    lstm_b2 = LSTM(300, return_sequences=True,activation='tanh', go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_b1)

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])
    concat_features = Dropout(0.4)(concat_features)

    #Dense layers
    #protein_features = Dense(600,activation='relu')(concat_features)
    #protein_features = Dropout(0.4)(protein_features)

    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)
    #protein_features_2 = Dense(300,activation='relu')(protein_features)
    #protein_features_2 = Dropout(0.4)(protein_features_2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(concat_features)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    #use Adam optimizer
    adam = Adam(lr=0.003)
    #Adam is fast, but tends to over-fit
    #SGD is low but gives great results, sometimes RMSProp works best, SWA can easily improve quality, AdaTune

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    # load_file = "./model/ac_LSTM_best_time_17.h5" # M: weighted_accuracy E: val_weighted_accuracy
    checkpoint_path = "gs://keras-python-models/checkpoints/" + str(datetime.date(datetime.now())) + ".h5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_loss', mode='min')


    return model

# def main(job_dir, **args):
# def main(job_dir, args):
def main(args):

    print('cwd from psp_lstm_gcp dir ', os.getcwd())

    # job_dir = os.environ["JOB_DIR"]
    job_dir = str(args.job_dir)
    logs_path = job_dir + 'logs/tensorboard'

    # with tf.device('/device:GPU:0'):
    #Load data
    print('job_dir' , job_dir)
    print(os.getcwd())
    # train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted_2()
    print(os.getcwd(), 'cwd before training data')
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted()
    # print(os.getcwd(), 'cwd before test data')
    #
    test_hot, testpssm, testlabel = load_cb513()


    #build model
    # model = build_model()
    # model = load_model('model_1.h5')
    #model = build_model_lstm()
    # model = load_model('model_1')
    #do google cloud authentication for outside users calling function
    #
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)

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
    #add callbacks for tesnorboard and history
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    # print(os.environ.get('BATCH_SIZE'))
    # print(os.environ.get('EPOCHS'))
    # batch_size = os.environ.get('BATCH_SIZE')
    # epochs = os.environ.get('EPOCHS')
    print(args.epochs)
    #batch_size = str(args(batch_size))
    #fit model
    print('Fitting model...')
    # history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
    # epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard,ReduceLROnPlateau()],shuffle=True)
    # # #

    # #save model locally and to google cloud bucket
    print('Saving model')
    save_path = 'model_lstm' + str(datetime.date(datetime.now())) + '.h5'
    model.save(save_path)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("keras-python-models")
    blob_path = 'models/model_lstm_hpconfig'+ str(datetime.date(datetime.now())) + '.h5'
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(save_path)

    # model.save('model_2.h5')
    # with open('model_1.h5', 'r') as f:
    #     with open('../models/model_1.h5', mode='w+', encoding="utf8", errors='ignore') as output_f:
    #         output_f.write(f.read())
        #blob.upload_from_file(f)
        #print('File uploaded')
    # #check if model name already exists
    # with file_io.FileIO('model_1.h5', mode='rb') as input_f:
    #      with file_io.FileIO(job_dir + '/models/model_1.h5', mode='w+') as output_f:
    #          output_f.write(input_f.read())
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='/Users/adammckenna/protein_structure_prediction_DeepLearning/psp_gcp_test_dir/service-account.json'
# These credentials will be used by any library that requests Application Default Credentials (ADC).
#Instead of using the GAC env variable, run # !gcloud auth application-default login,
#this will authenticate google account with google sdk, crdenetials will be automatically stored in config file which gcloud sdk looks for credentials


    # storage_client = storage.Client()
    # bucket = storage_client.get_bucket("keras-python-models")
    # blob = bucket.blob('models/models_LSTM_100.h5')
    # blob.upload_from_filename('model_2.h5')

    # plot_model(history)

def plot_model(history):

    #plot train and validation accuracy on history
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')
    plt.show()
    plt.close()

    #plot train and validation loss on history
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.show()
    plt.close()

    # plt.figure()
    # plt.plot(history.history['mean_absolute_error'])
    # plt.plot(history.history['val_mean_absolute_error'])
    # plt.title('Model Mean Absolute Error')
    # plt.ylabel('Mean Absolute Error')
    # plt.xlabel('Epoch')
    # plt.legend(['train_mae', 'val_mae'], loc='upper left')
    # plt.show()
    # plt.close()



#with open(path, encoding="utf8", errors='ignore') as f:

# ##Running the app
# if __name__ == "__main__":

print('Something')
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=42,
                    help='batch size for training data (default: 42)')
# parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
#                     help='input batch size for testing (default: 1024)')
parser.add_argument('--data_dir', type=str, default='../data',
                    help='Directory for training and test datasets')
parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
                    help='Google Storage Bucket storing data and logs')
# parser.add_argument('--result_dir', type=str, default='./result',
#                     help='Output directory (default: ./result)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
                    help ='The number of nodes for first LSTM hidden layer')
parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
                    help ='The number of nodes for second LSTM hidden layer')
parser.add_argument('-dr', '--dropout', type=float, default = 0.5,
                    help='The dropout applied to input (default = 0.5)')
parser.add_argument('-op', '--optimizer', default = 'adam',
                    help='The optimizer used in compiling and fitting the models')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')
parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = 'gs://keras-python-models')
args = parser.parse_args()
print(args.epochs)
# parser = argparse.ArgumentParser()
#
# # Input Arguments
# parser.add_argument(
#   '--job-dir',
#   help='GCS location to write checkpoints and export models',
#   required=True
# )
# args = parser.parse_args()
# arguments = args.__dict__
#

# main(args.job_dir, args)
# args = args.__dict__
# subprocess.call(["../psp_gcp/gcp_deploy.sh"],shell =True)

main(args)
# gcp shell script calls the psp_lstm_gcp script, does not have to be a main function ??
