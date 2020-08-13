#PSP model using the Keras functional API with a LSTM neural network

#Importing libraries and dependancies required for building the model
import numpy as np
import gzip
import h5py
import tensorflow as tf
#from tensorflow import keras
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard
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
from training.get_dataset import *

# print(os.getcwd())
# d = os.getcwd()
# g = d + '/' + 'training/psp_keras_training.json'
# print(g)
# storage_client = storage.Client.from_service_account_json(psp_keras_training.json)
# bucket = storage_client.get_bucket("keras-python-models")
# filename = "model_1.h5"
# # blob = bucket.blob(filename)
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
config_proto.graph_options.rewrite_options.memory_optimization  = off
# session = tf.Session(config=config_proto)
session = tf.compat.v1.Session(config=config_proto)

set_session(session)

#turn into a class and have hyperparameters as class arguments
#Function to build the model
def build_model_lstm_hpconfig(args):


    conv2d_layer1_filters=int(args.conv2d_layer1_filters)
    conv2d_layer2_filters=int(args.conv2d_layer2_filters)
    conv2d_activation=args.conv2d_activation
    conv2d_dropout=float(args.conv2d_dropout)
    lstm_layer1_nodes=int(args.lstm_layers1)
    lstm_layer2_nodes=int(args.lstm_layers2)
    lstm_dropout=float(args.lstm_dropout)
    lstm_recurrent_dropout=float(args.lstm_recurrent_dropout)
    after_lstm_dropout=float(args.after_lstm_dropout)
    optimizer=args.optimizer
    learning_rate = float(args.learning_rate)

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
    conv1_features = Conv1D(conv2d_layer1_filters,1,strides=1,activation=conv2d_activation, padding='same')(concat)
    max_pool_1d = MaxPooling1D(pool_size = 2, strides =1, padding='same')(conv1_features)
    conv1_features = Reshape((700, 42, 1))(max_pool_1d)

    # print ('conv1_features shape', conv1_features.get_shape())
    #print ('max_pool_1d shape', max_pool_1d.get_shape())

    #reshape 1D convolutional layer
    conv1_features = Reshape((700, 42, 1))(max_pool_1d)

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(conv2d_layer1_filters,3,strides=1,activation=conv2d_activation, padding='same')(conv1_features)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(conv2d_dropout)(max_pool_2D)
    print ('MaxPooling Shape', max_pool_2D.get_shape())

    #2D Convolutional layers and 2D Max Pooling
    conv2_features = Conv2D(conv2d_layer2_filters,3,strides=1,activation='relu', padding='same')(max_pool_2D)
    max_pool_2D = MaxPooling2D(pool_size=(2,2), strides=1, padding ='same')(conv2_features)
    max_pool_2D = Dropout(conv2d_dropout)(max_pool_2D)
    print ('Conv2D layer1 shape',conv2_features.get_shape())

    #conv2_batch_normal = BatchNormalization()(conv2_features)
    #max_pool_2D = MaxPooling2D(pool_size=(1, 2), strides=None, border_mode='same')(conv2_batch_normal)

    #conv2_features = Convolution2D(84,3,1,activation='relu', padding='same')(conv2_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()

    #reshape 2D convolutional layers
    conv2_features = Reshape((700, 84*42))(max_pool_2D)

    #conv2_features = Dense(500, activation='relu')(conv2_features)

    #Long Short Term Memory layers with tanh activation, sigmoid recurrent activiation and dropout of 0.5
    lstm_f1 = LSTM(lstm_layer1_nodes,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=lstm_dropout,recurrent_dropout=lstm_recurrent_dropout)(conv2_features)
    lstm_b1 = LSTM(lstm_layer1_nodes, return_sequences=True, activation='tanh',go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=lstm_recurrent_dropout)(conv2_features)

    lstm_f2 = LSTM(lstm_layer2_nodes, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=lstm_dropout,recurrent_dropout=lstm_recurrent_dropout)(lstm_f1)
    lstm_b2 = LSTM(lstm_layer2_nodes, return_sequences=True,activation='tanh', go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=lstm_recurrent_dropout)(lstm_b1)

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])
    concat_features = Dropout(after_lstm_dropout)(concat_features)

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

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = optimizer, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    # load_file = "./model/ac_LSTM_best_time_17.h5" # M: weighted_accuracy E: val_weighted_accuracy
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_loss', mode='min')
    checkpoint_path = "gs://keras-python-models/checkpoints/" + str(datetime.date(datetime.now())) + ".h5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')
#     checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_loss', mode='min')

    #print 'test loss:', score[0]
    #print 'test accuracy:', score[1]
    return model

def main(args):
    print('cwd from psp_lstm_gcp dir ', os.getcwd())

    # job_dir = os.environ["JOB_DIR"]
    job_dir = str(args.job_dir)
    logs_path = job_dir + 'logs/tensorboard'
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)


    # with tf.device('/device:GPU:0'):
    #Load data
    print('job_dir' , job_dir)
    print(os.getcwd())
    # train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted_2()
    print(os.getcwd(), 'cwd before training data')
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(0.25)
    test_hot, testpssm, testlabel = load_cb513(0.25)

    #model = build_model_lstm_hpconfig(args[1: 10 ])....

    model = build_model_lstm_hpconfig(args)

    print('Fitting model...')
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
    epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],shuffle=True)

    # #save model
    print('Saving model')
    save_path = 'model_lstm_hpconfig' + str(datetime.date(datetime.now())) + '.h5'
    model.save(save_path)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("keras-python-models")
    blob_path = 'models/model_lstm_hpconfig'+ str(datetime.date(datetime.now())) + '.h5'
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(save_path)

    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=2,batch_size=1)
    eval_score = score[1]
    print('Model Loss : ', score[0])
    print('Model Accuracy : ', eval_score)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='eval_score',
        metric_value=eval_score,
        global_step=1000
    )

print('Something')
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=42,
                    help='batch size for training data (default: 42)')
# parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
#                     help='input batch size for testing (default: 1024)')
parser.add_argument('--data_dir', type=str, default='.',
                    help='Directory for training and test datasets')
parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
                    help='Google Storage Bucket storing data and logs')
# parser.add_argument('--result_dir', type=str, default='./result',
#                     help='Output directory (default: ./result)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
                    help ='The number of nodes for first LSTM hidden layer')

parser.add_argument('-conv2D_1', '--conv2d_layer1_filters', type=int, default=42,
                    help ='The number of filters for first Conv2D hidden layer')

parser.add_argument('-conv2D_2', '--conv2d_layer2_filters', type=int, default=84,
                    help ='The number of filters for second Conv2D hidden layer')

parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
                    help ='The number of nodes for second LSTM hidden layer')

parser.add_argument('-lstm_dropout', '--lstm_dropout', default=0.5,
                    help ='Dropout for LSTM Layers')

parser.add_argument('-lstm_recurrent_dropout', '--lstm_recurrent_dropout', default=0.5,
                    help ='Recurrent Dropout for LSTM Layers')

parser.add_argument('-conv2d_dropout', '--conv2d_dropout', type=float, default = 0.5,
                    help='The dropout applied to Conv2D layers input (default = 0.5)')

parser.add_argument('-op', '--optimizer', default = 'adam',
                    help='The optimizer used in compiling and fitting the models')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')

parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = 'gs://keras-python-models')

parser.add_argument('-conv2d_activation', '--conv2d_activation', help='Activation function applied to Conv2D layers',required=False,
                    default = 'relu')

parser.add_argument('-after_lstm_dropout', '--after_lstm_dropout', help='Dropout applied after LSTM layers',required=False,
                    type=int, default = 0.5)


parser.add_argument('-lr', '--learning_rate', help='Learning rate for training model',required=False, default = .003)

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
