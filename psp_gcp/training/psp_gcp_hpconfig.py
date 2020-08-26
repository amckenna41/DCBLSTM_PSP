#PSP model using the Keras functional API with a LSTM neural network

#Importing libraries and dependancies required for building the model
import numpy as np
import gzip
import h5py
import tensorflow as tf
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard
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
import hypertune
from training.gcp_utils import *
from training.get_dataset import *
from training.plot_model import *

tf.compat.v1.reset_default_graph()
from tensorflow.core.protobuf import rewriter_config_pb2
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.compat.v1.keras.backend import set_session

# config_proto = tf.ConfigProto()
config_proto = tf.compat.v1.ConfigProto()

off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
# config_proto.graph_options.rewrite_options.memory_optimization  = off
# session = tf.Session(config=config_proto)
session = tf.compat.v1.Session(config=config_proto)

set_session(session)

BUCKET_PATH = "gs://keras-python-models"

#building neural network with hyperparameters passed in upon execution of the
#gcp_hptuning script
def build_model_hpconfig(args):

    #parsing and assigning hyperparameter variables from argparse
    conv2d_layer1_filters=int(args.conv2d_layer1_filters)
    conv2d_layer2_filters=int(args.conv2d_layer2_filters)
    conv2d_activation=args.conv2d_activation
    conv2d_dropout=float(args.conv2d_dropout)
    recurrent_layer1 = int(args.recurrent_layer1)
    recurrent_layer2 = int(args.recurrent_layer2)
    recurrent_dropout = float(args.recurrent_dropout)
    after_recurrent_dropout = float(args.after_recurrent_dropout)
    recurrent_recurrent_dropout = float(args.recurrent_recurrent_dropout)
    optimizer=args.optimizer
    learning_rate = float(args.learning_rate)
    bidirection = args.bidirection
    recurrent_layer = args.recurrent_layer

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(700,21), name='aux_input')
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)

    #concatenate input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    #reshape 1D convolutional layer
    conv1_features = Reshape((700, 42, 1))(concat)


    #both conv2d layers need to be same amount of filters

    ###Hidden Layers###
    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(conv2d_layer1_filters,3,strides=1,activation=conv2d_activation, padding='same')(conv1_features)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(conv2d_dropout)(max_pool_2D)
    print ('Conv2D Shape after MaxPooling', max_pool_2D.get_shape())

    #batch BatchNormalization between Conv Layers
    batch_norm = BatchNormalization()(max_pool_2D)

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(conv2d_layer2_filters,3,strides=1,activation=conv2d_activation, padding='same')(batch_norm)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(conv2d_dropout)(max_pool_2D)
    print ('Conv2D Shape after MaxPooling', max_pool_2D.get_shape())

    batch_norm = BatchNormalization()(max_pool_2D)

    #reshape 2D convolutional layers
    # conv2_features = Reshape((700, 42*42))(max_pool_2D)
    conv2_features = Reshape((700, conv2d_layer1_filters*conv2d_layer2_filters))(batch_norm)


    ######## Recurrent Layers ########
    if (recurrent_layer == 'lstm'):
        if (bidirection):

            #Creating Bidirectional LSTM layers
            lstm_f1 = Bidirectional(LSTM(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout, recurrent_dropout=recurrent_recurrent_dropout))(conv2_features)
            lstm_f2 = Bidirectional(LSTM(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout))(lstm_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, conv2_features])
            # concat_features = Dropout(after_recurrent_dropout)(concat_features)


        else:
            #Creating unidirectional LSTM Layers
            lstm_f1 = LSTM(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout)(conv2_features)

            lstm_f2 = LSTM(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout)(lstm_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, conv2_features])
            # concat_features = Dropout(after_recurrent_dropout)(concat_features)


    elif (recurrent_layer == 'gru'):
        if (bidirection):

            #Creating Bidirectional GRU layers
            gru_f1 = Bidirectional(GRU(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout))(conv2_features)

            gru_f2 = Bidirectional(GRU(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout))(gru_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv2_features])
            # concat_features = Dropout(after_recurrent_dropout)(concat_features)


        else:
            #Creating unidirectional GRU Layers
            gru_f1 = GRU(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout)(conv2_features)

            gru_f2 = GRU(recurrent_layer1, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout)(gru_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv2_features])
            # concat_features = Dropout(after_recurrent_dropout)(concat_features)
    else:

        print('Only LSTM and GRU recurrent layers are used in this model')
        return
###################


    #Dense fully-connected layers
    #protein_features = Dense(600,activation='relu')(concat_features)
    #protein_features = Dropout(0.4)(protein_features)

    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)
    #protein_features_2 = Dense(300,activation='relu')(protein_features)
    #protein_features_2 = Dropout(0.4)(protein_features_2)
    # concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv2_features])

    #Final Output layer with 8 nodes for the 8 output classifications
    # main_output = Dense(8, activation='softmax', name='main_output')(concat_features)
    main_output = Dense(8, activation='softmax', name='main_output')(concat_features)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #Set optimizer to be used with the model, default is Adam
    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate, name='adam')
    elif optimizer == 'sgd':
        optimizer = SGD(lr=0.01, momentum=0.0, nestero=False, name='SGD')
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate, centered = True, name='RMSprop')
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate = learning_rate, name='Adagrad')
    elif optimizer == 'adamax':
        optimizer = Adamax(learning_rate=learning_rate, name='Adamax')
    else:
        optimizer = 'adam'
        optimizer = Adam(lr=learning_rate, name='adam')

    #Nadam & Ftrl optimizers

    #use Adam optimizer
    #optimizer = Adam(lr=0.003)
    #Adam is fast, but tends to over-fit
    #SGD is low but gives great results, sometimes RMSProp works best, SWA can easily improve quality, AdaTune

    #compile model using optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = optimizer, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    #get summary of model including its layers and num parameters
    model.summary()

    #set early stopping and checkpoints for model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = BUCKET_PATH + "/checkpoints/" + str(datetime.date(datetime.now())) +\
        '_' + str((datetime.now().strftime('%H:%M'))) + ".h5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')

    return model

def main(args):

    job_dir = str(args.job_dir)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    all_data = float(args.alldata)

    all_data = 0.01
    epochs = 1
    # logs_path = job_dir + 'logs/tensorboard/' + 'hp_config_'+str(datetime.date(datetime.now()))
    logs_path = str(args.logs_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    print('TensorBoard logs stored at: ', logs_path)
    # with tf.device('/device:GPU:0'):
    #Load data

    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)
    test_hot, testpssm, testlabel = load_cb513(all_data)

    # model = build_model_lstm_hpconfig(args[1: 10 ])....

    model = build_model_hpconfig(args)

    print('Fitting model...')
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],shuffle=True)

    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,
            batch_size=1,callbacks=[tensorboard])
    #evaluate with casp10 and casp11 test datasets

    eval_score = score[1]
    print('Model Loss : ', score[0])
    print('Model Accuracy : %.2f%% ', (eval_score*100))

    #evaluation storage metadata

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='eval_score',
        metric_value=eval_score,
        global_step=1000
    )

    model_save_path = 'model_hptuning_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '_accuracy-'+ str(score[1]) \
        +'_loss-' + str(score[0]) + '.h5'

    upload_history(history,model_save_path,score)
    upload_model(model, args,model_save_path)
    plot_history(history.history, show_histograms=True, show_boxplots=True, show_kde=True)



print('Something')
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=120,
                    help='batch size for training data (default: 42)')
parser.add_argument('--data_dir', type=str, default='.',
                    help='Directory for training and test datasets')
parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
                    help='Google Storage Bucket storing data and logs')
# parser.add_argument('--result_dir', type=str, default='./result',
#                     help='Output directory (default: ./result)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('-recurrent_layer1', '--recurrent_layer1', type=int, default=400,
                    help ='The number of nodes for first recurrent hidden layer')

parser.add_argument('-conv2D_1', '--conv2d_layer1_filters', type=int, default=42,
                    help ='The number of filters for first Conv2D hidden layer')

parser.add_argument('-conv2D_2', '--conv2d_layer2_filters', type=int, default=42,
                    help ='The number of filters for second Conv2D hidden layer')



parser.add_argument('-conv_layers', '--conv_layers', type=int, default=2,
                    help ='The number of convolutional layers before the recurrent layers')


parser.add_argument('-recurrent_layer2', '--recurrent_layer2', type=int, default=300,
                    help ='The number of nodes for second recurrent hidden layer')

parser.add_argument('-recurrent_dropout', '--recurrent_dropout', default=0.5,
                    help ='Dropout for recurrent hidden Layers')

parser.add_argument('-recurrent_recurrent_dropout', '--recurrent_recurrent_dropout', default=0.5,
                    help ='Recurrent Dropout for recurrent hidden Layers')

parser.add_argument('-conv2d_dropout', '--conv2d_dropout', type=float, default = 0.5,
                    help='The dropout applied to Conv2D layers input (default = 0.5)')

parser.add_argument('-op', '--optimizer', default = 'adam',
                    help='The optimizer used in compiling and fitting the models')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')

parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = BUCKET_PATH)

parser.add_argument('-conv2d_activation', '--conv2d_activation', help='Activation function applied to Conv2D layers',required=False,
                    default = 'relu')

parser.add_argument('-after_recurrent_dropout', '--after_recurrent_dropout', help='Dropout applied after recurrent layers',required=False,
                    type=float, default = 0.5)

parser.add_argument('-alldata', '--alldata', type =float, default=0.25,
                    help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc, Default: 0.25')
#validation on all_dat
parser.add_argument('-bidirection', '--bidirection', type =bool, default=True,
                    help='Select whether you want the LSTM to be unidirectional or bidirectional - True = BDLSTM, False - ULSTM, default: True')
#validation on all_dat
parser.add_argument('-batch_norm', '--batch_norm', type =bool, default=True,
                    help='Select whether you want a BatchNormalization layer b/w and after Convolutional layers default: True')

parser.add_argument('-recurrent_layer', '--recurrent_layer',  type = str.lower, default='lstm',
                    help='Select what recurrent layer to use in network - GRU or LSTM, default: LSTM')

parser.add_argument('-lr', '--learning_rate', help='Learning rate for training model',required=False, default = .003)

parser.add_argument('-logs_dir', '--logs_dir', help='Directory on cloud storage for Tensorboard logs',required=False, default = (BUCKET_NAME + "/logs/tensorboard"))

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





#
#
#
#
# parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
# parser.add_argument('-b', '--batch_size', type=int, default=42,
#                     help='batch size for training data (default: 42)')
# # parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
# #                     help='input batch size for testing (default: 1024)')
# parser.add_argument('--data_dir', type=str, default='.',
#                     help='Directory for training and test datasets')
# parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
#                     help='Google Storage Bucket storing data and logs')
# # parser.add_argument('--result_dir', type=str, default='./result',
# #                     help='Output directory (default: ./result)')
# # parser.add_argument('--seed', type=int, default=1, metavar='S',
# #                     help='random seed (default: 1)')
# parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
#                     help ='The number of nodes for first LSTM hidden layer')
#
# parser.add_argument('-conv2D_1', '--conv2d_layer1_filters', type=int, default=42,
#                     help ='The number of filters for first Conv2D hidden layer')
#
# parser.add_argument('-conv2D_2', '--conv2d_layer2_filters', type=int, default=84,
#                     help ='The number of filters for second Conv2D hidden layer')
#
# parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
#                     help ='The number of nodes for second LSTM hidden layer')
#
# parser.add_argument('-lstm_dropout', '--lstm_dropout', default=0.5,
#                     help ='Dropout for LSTM Layers')
#
# parser.add_argument('-lstm_recurrent_dropout', '--lstm_recurrent_dropout', default=0.5,
#                     help ='Recurrent Dropout for LSTM Layers')
#
# parser.add_argument('-conv2d_dropout', '--conv2d_dropout', type=float, default = 0.5,
#                     help='The dropout applied to Conv2D layers input (default = 0.5)')
#
# parser.add_argument('-op', '--optimizer', default = 'adam',
#                     help='The optimizer used in compiling and fitting the models')
#
# parser.add_argument('-e', '--epochs', type=int, default=10,
#                     help='The number of epochs to run on the model')
#
# parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
#                     default = 'gs://keras-python-models')
#
# parser.add_argument('-conv2d_activation', '--conv2d_activation', help='Activation function applied to Conv2D layers',required=False,
#                     default = 'relu')
#
# parser.add_argument('-after_lstm_dropout', '--after_lstm_dropout', help='Dropout applied after LSTM layers',required=False,
#                     type=int, default = 0.5)
# parser.add_argument('-alldata', '--alldata', type =float, default=0.25,
#                     help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc, Default: 0.25')
# #validation on all_dat
# parser.add_argument('-bidirection', '--bidirection', type =bool, default=True,
#                     help='Select whether you want the LSTM to be unidirectional or bidirectional - True = BDLSTM, False - ULSTM, default: True')
#
# parser.add_argument('-recurrent_layer', '--recurrent_layer',  type = str.lower, default='lstm',
#                     help='Select what recurrent layer to use in network - GRU or LSTM, default: LSTM')
#
# parser.add_argument('-lr', '--learning_rate', help='Learning rate for training model',required=False, default = .003)
#
# args = parser.parse_args()
