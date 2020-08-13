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
import json
from google.oauth2 import service_account
from tensorflow.core.protobuf import rewriter_config_pb2
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.compat.v1.keras.backend import set_session

# config_proto = tf.ConfigProto()
config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
# session = tf.Session(config=config_proto)
session = tf.compat.v1.Session(config=config_proto)

set_session(session)

BUCKET_NAME = "keras-python-models"
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'


def build_gru_model():

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
    gru_f1 = GRU(400,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)
    gru_b1 = GRU(400, return_sequences=True, activation='tanh',go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)

    gru_f2 = GRU(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(gru_f1)
    gru_b2 = GRU(300, return_sequences=True,activation='tanh', go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(gru_b1)

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=-1)([gru_f2, gru_b2, conv2_features])
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
    test_hot, testpssm, testlabel = load_cb513()
    #build model

    model = build_gru_model()

    #convert batch_size and epochs from str to int
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
    # #save model
    print('Saving model')
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

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')
    plt.close()


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
