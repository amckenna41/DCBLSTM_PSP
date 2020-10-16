#PSP model using BLSTM RNN with CNN + DNN

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Input, Conv1D, Embedding, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
import os
import sys
from datetime import date
from datetime import datetime


#set required parameters and configuration for TensorBoard
tf.compat.v1.reset_default_graph()
from tensorflow.core.protobuf import rewriter_config_pb2
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.compat.v1.keras.backend import set_session
# config_proto = tf.ConfigProto()
config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.gpu_options.allow_growth = True
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
# session = tf.Session(config=config_proto)
session = tf.compat.v1.Session(config=config_proto)
set_session(session)

#building BLSTM_3xConv_Model
def build_model():

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(700,21), name='aux_input')

    #get shape of input layers
    print ("Protein Sequence shape: ", main_input.get_shape())
    print ("Protein Profile shape: ",auxiliary_input.get_shape())

    #concatenate 2 input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    #3x1D Convolutional Hidden Layers with BatchNormalization, Dropout and MaxPooling
    conv_layer1 = Conv1D(16, 7, kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv2D_act)
    max_pool_1D_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    conv_layer2 = Conv1D(32, 7, padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer2)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv2D_act)
    max_pool_1D_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    conv_layer3 = Conv1D(64, 7,kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer3)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv2D_act)
    max_pool_1D_3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    #concatenate convolutional layers
    conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])

     ######## Recurrent Bi-Directional Long-Short-Term-Memory Layers ########
    lstm_f1 = Bidirectional(LSTM(200,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(conv_features)

    lstm_f2 = Bidirectional(LSTM(200, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(lstm_f1)

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, conv_features])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    dense_1 = Dense(600, activation='relu')(conv_features)
    dense_1_dropout = Dropout(0.3)(dense_1)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(dense_4_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.0015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])
    model.summary()

    return model
