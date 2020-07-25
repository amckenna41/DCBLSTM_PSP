#PSP model using the Keras functional API with a LSTM neural network

#Importing libraries and dependancies required for building the model
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, GlobalMaxPooling1D, TimeDistributed, Embedding, MaxPooling1D, LSTM, Dense, merge, Conv1D, Conv2D, Convolution2D, GRU, Concatenate, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime

#Function to build the model
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

    #1D Convolutional layer
    conv1_features = Conv1D(42,1,strides=1,activation='relu', padding='same')(concat)
    #max_pool_1d = MaxPooling1D()(conv1_features)
    #max_pool_1d = GlobalMaxPooling1D()(conv1_features)
    #max_pool_1d = MaxPooling1D(padding="valid")(conv1_features)

    # print 'conv1_features shape', conv1_features.get_shape()
    conv1_features = Reshape((700, 42, 1))(conv1_features)


    #output_shape = (input_shape - pool_size + 1) / strides)

    #conv2_features = Convolution2D(42,3,strides=(1,1),activation='relu', padding='same')(conv1_features)
    conv2_features = Conv2D(42,3,strides=(1,1),activation='relu', padding='same')(conv1_features)
    conv2_batch_normal = BatchNormalization()(conv2_features)
    conv2_features = Conv2D(84,3,strides=(1,1),activation='relu', padding='same')(conv2_batch_normal)
    conv2_batch_normal = BatchNormalization()(conv2_features)
    max_pool_2D = MaxPooling2D(pool_size=(1, 2), strides=None, border_mode='same')(conv2_batch_normal)

    #tf.keras.layers.MaxPooling2D(
    #pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs


    #conv2_features = Convolution2D(84,3,1,activation='relu', padding='same')(conv2_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()

    conv2_features = Reshape((700,42*42))(max_pool_2D)
    conv2_features = Dropout(0.5)(conv2_features)
    conv2_features = Dense(500, activation='relu')(conv2_features)

    #, activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5
    lstm_f1 = LSTM(400,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)
    lstm_b1 = LSTM(400, return_sequences=True, activation='tanh',go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)

    lstm_f2 = LSTM(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_f1)
    lstm_b2 = LSTM(300, return_sequences=True,activation='tanh', go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_b1)

    #concat_features = merge([lstm_f2, lstm_b2, conv2_features], mode='concat', concat_axis=-1)
    concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])
    concat_features = Dropout(0.4)(concat_features)
    protein_features = Dense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)
    protein_features_2 = Dense(300,activation='relu')(protein_features)
    main_output = Dense(8, activation='softmax', name='main_output')(protein_features_2)
    #main_output = TimeDistributedt()(protein_features)


    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    adam = Adam(lr=0.003)
    #Adam is fast, but tends to over-fit
    #SGD is low but gives great results, sometimes RMSProp works best, SWA can easily improve quality, AdaTune
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    model.summary()

    # print "####### look at data's shape#########"
    # print traindatahot.shape, trainpssm.shape, trainlabel.shape, testdatahot.shape, testpssm.shape,testlabel.shape, valdatahot.shape,valpssm.shape,vallabel.shape
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    ######################
    # load_file = "./model/ACNN/acnn1-3-42-400-300-blstm-FC600-42-cb6133F-0.5-0.4.h5"
    #################################
    # load_file = "./model/ac_LSTM_best_time_17.h5" # M: weighted_accuracy E: val_weighted_accuracy
#     checkpoint_path = "./model/deepRNN_LSTM_best" + str(datetime.date(datetime.now())) + ".h5"
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_loss', mode='min')


    return model
