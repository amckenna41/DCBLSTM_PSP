#Create LSTM model using just PSSP data as input into the network

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


def build_lstm_pssm_model():

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,21), dtype='float32', name='main_input')

    #get shape of input layers
    print (main_input.get_shape())

    #concatenate input layers
    concat = main_input

    # #1D Convolutional layer and maxpooling to downsample features
    # conv1_features = Conv1D(42,1,strides=1,activation='relu', padding='same')(concat)
    # max_pool_1d = MaxPooling1D(pool_size = 2, strides =1, padding='same')(conv1_features)

    # print ('conv1_features shape', conv1_features.get_shape())
    #print ('max_pool_1d shape', max_pool_1d.get_shape())

    #reshape 1D convolutional layer
    conv1_features = Reshape((700, 42, 1))(concat)

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(conv1_features)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(0.5)(max_pool_2D)
    print ('MaxPooling Shape', max_pool_2D.get_shape())

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(84,3,strides=1,activation='relu', padding='same')(max_pool_2D)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    # conv2_features = Conv2D(42,3,strides=1,activation='relu', padding='same')(max_pool_2D)

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
    model = Model(inputs=[main_input], outputs=[main_output])
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
    #checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_loss', mode='min')


    return model
