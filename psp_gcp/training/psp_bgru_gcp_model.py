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
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Input, Conv1D, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
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
import pickle
import matplotlib.pyplot as plt
from training.get_dataset import *
from training.plot_model import *
from training.gcp_utils import *

import json
from google.oauth2 import service_account
tf.compat.v1.reset_default_graph()

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
storage_client = storage.Client()
bucket = storage_client.get_bucket("keras-python-models")
# with tf.compat.v1.Session(config=config_proto) as sess:
#     set_session(sess)
#     # or creating the writer inside the session
#     writer = tf.summary.FileWriter(BUCKET_NAME + '/logs/tensorboard', sess.graph)


BUCKET_PATH = "gs://keras-python-models"
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'

#building BGRU_3xConv_Model
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

#     #1D Convolutional layer and maxpooling to downsample features
#     conv1_features = Conv1D(42,1,strides=1,activation='relu', padding='same')(concat)
#     max_pool_1d = MaxPooling1D(pool_size = 2, strides =1, padding='same')(conv1_features)
#     conv1_features = Reshape((700, 42, 1))(max_pool_1d)

    # print ('conv1_features shape', conv1_features.get_shape())
    #print ('max_pool_1d shape', max_pool_1d.get_shape())

    #reshape 1D convolutional layer
    conv1_features = Reshape((700, 42, 1))(concat)


    #2D Convolutional layers and 2D Max Pooling
#     conv2_features = Conv2D(84,3,strides=1,activation='relu', padding='same')(max_pool_2D)
    conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(conv1_features)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    #batch BatchNormalization between Conv Layers
    batch_norm = BatchNormalization()(max_pool_2D)

    conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(batch_norm)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('Conv2D layer2 shape',conv2_features.get_shape())

    #batch BatchNormalization between Conv Layers
    batch_norm = BatchNormalization()(max_pool_2D)

    conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(batch_norm)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('Conv2D layer3 shape',conv2_features.get_shape())

    #batch BatchNormalization between Conv Layers
    batch_norm = BatchNormalization()(max_pool_2D)

    #reshape 2D convolutional layers
    conv2_features = Reshape((700, 42*42))(batch_norm)
    #conv2_features = Reshape((700, 42*42))(max_pool_2D)

    #conv2_features = Dense(500, activation='relu')(conv2_features)

    # ######## Recurrent Layers ########
    gru_f1 = Bidirectional(GRU(400,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(conv2_features)

    gru_f2 = Bidirectional(GRU(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(gru_f1)


    #concatenate GRU with convolutional layers
    concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv2_features])

    # concat_features = Dropout(0.4)(concat_features)

    #Dense layers
    #protein_features = Dense(600,activation='relu')(concat_features)
    #protein_features = Dropout(0.4)(protein_features)

    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)
    #protein_features_2 = Dense(300,activation='relu')(protein_features)
    #protein_features_2 = Dropout(0.4)(protein_features_2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    # main_output = Dense(8, activation='softmax', name='main_output')(concat_features)
    main_output = Dense(8, activation='softmax', name='main_output')(concat_features)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    #use Adam optimizer
    adam = Adam(lr=0.003)
    #Adam is fast, but tends to over-fit
    #SGD is low but gives great results, sometimes RMSProp works best, SWA can easily improve quality, AdaTune

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = "/gru_3conv_" + str(datetime.date(datetime.now())) + ".h5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')

    return model

def main(args):

    #setting parsed input arguments
    job_dir = str(args.job_dir)
    all_data = float(args.alldata)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)

    # logs_path = job_dir + 'logs/tensorboard/' + str(datetime.date(datetime.now()))  + \
    #     '_' + str((datetime.now().strftime('%H:%M')))
    print("Logs Path: ", logs_path)
    # with tf.device('/device:GPU:0'):
    #Load data
    print('job_dir' , job_dir)

    print('Running model using {}%% of data'.format(int(all_data*100)))
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)
    test_hot, testpssm, testlabel = load_cb513(all_data)


    #build model
    print('Building 2conv BGRU model')
    model = build_model()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    print(args.epochs)
    #fit model
    print('Fitting model...')
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard],shuffle=True)
    # ReduceLROnPlateau()-callback

    #Saving pickle of history so that it can later be used for visualisation of the model

    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,batch_size=1)
    # eval_score = score[1]

    loss_summary = tf.summary.scalar(name='Loss Summary', data=score[0])
    accuracy_summary = tf.summary.scalar(name='Accuracy Summary', data=score[1])

    print('Model Loss : ', score[0])
    print('Model Accuracy : ', score[1])


    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='/Users/adammckenna/protein_structure_prediction_DeepLearning/psp_gcp_test_dir/service-account.json'
# These credentials will be used by any library that requests Application Default Credentials (ADC).
#Instead of using the GAC env variable, run # !gcloud auth application-default login,
#this will authenticate google account with google sdk, crdenetials will be automatically stored in config file which gcloud sdk looks for credentials

    model_save_path = 'model_bgru_3conv_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '_accuracy-'+ str(score[1]) \
        +'_loss-' + str(score[0]) + '.h5'

    upload_history(history,model_save_path,score)
    upload_model(model, args,model_save_path)
    plot_history(history.history, show_histograms=True, show_boxplots=True, show_kde=True)


parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=42,
                    help='batch size for training data (default: 42)')
parser.add_argument('-sb','--storage_bucket', type=str, default=BUCKET_PATH,
                    help='Google Storage Bucket storing data and logs')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')
parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = BUCKET_PATH)
parser.add_argument('-alldata', '--alldata', type =float, default=1,
                    help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc')
parser.add_argument('-logs_dir', '--logs_dir', help='Directory on cloud storage for Tensorboard logs',required=False, default = (BUCKET_NAME + "/logs/tensorboard"))
#validation on all_data
args = parser.parse_args()


main(args)
