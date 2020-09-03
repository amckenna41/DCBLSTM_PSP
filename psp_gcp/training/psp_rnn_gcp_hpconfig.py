#PSP model using the Keras functional API, using CNN + RNN + DNN

#import required modules and dependancies
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Input, Conv1D, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Adagrad
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
import pandas as pd
# from io import BytesIO
# from tensorflow.python.lib.io import file_io
import os
import sys
from datetime import date
from datetime import datetime
import hypertune
from training.training_utils.get_dataset import *
from training.training_utils.plot_model import *
from training.training_utils.gcp_utils import *

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

#initialise bucket and GCP storage client
BUCKET_PATH = "gs://keras-python-models-2"
BUCKET_NAME = "keras-python-models-2"
# storage_client = storage.Client()
# bucket = storage_client.get_bucket(BUCKET_NAME)
# TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'

#building neural network with hyperparameters passed in upon execution of the
#gcp_hptuning script
def build_model_hpconfig(args):

    #parsing and assigning hyperparameter variables from argparse
    conv1_filters=int(args.conv1_filters)
    conv2_filters=int(args.conv2_filters)
    conv3_filters=int(args.conv3_filters)
    window_size=int(args.window_size)
    kernel_regularizer = args.kernel_regularizer
    # conv_dropout=(args.conv2d_dropout)
    max_pool_size = int(args.pool_size)
    conv2d_activation=args.conv2d_activation
    conv2d_dropout=float(args.conv2d_dropout)
    conv1d_initializer = args.conv_weight_initializer
    recurrent_layer1 = int(args.recurrent_layer1)
    recurrent_layer2 = int(args.recurrent_layer2)
    recurrent_dropout = float(args.recurrent_dropout)
    after_recurrent_dropout = float(args.after_recurrent_dropout)
    recurrent_recurrent_dropout = float(args.recurrent_recurrent_dropout)
    recurrent_initalizer = args.recurrent_weight_initializer
    optimizer=args.optimizer
    learning_rate = float(args.learning_rate)
    bidirection = args.bidirection
    recurrent_layer = str(args.recurrent_layer)
    dense_dropout = float(args.dense_dropout)
    dense_1 = int(args.dense_1)
    dense_2 = int(args.dense_2)
    dense_3 = int(args.dense_3)
    dense_4 = int(args.dense_4)
    dense_initializer = args.dense_weight_initializer

    print('BIDIRECTION:', bidirection)
    print('RECURRENT_LAYER:', recurrent_layer)

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(700,21), name='aux_input')

    #get shape of input layers
    print ("Protein Sequence shape: ", main_input.get_shape())
    print ("Protein Profile shape: ",auxiliary_input.get_shape())

    # print(window_size,conv1d_initializer, conv_dropout, max_pool_size)
    #concatenate input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    #3x1D Convolutional Hidden Layers with BatchNormalization and MaxPooling
    conv_layer1 = Conv1D(conv1_filters, window_size, kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(conv2d_dropout)(conv2D_act)
    # ave_pool_1 = AveragePooling1D(2, 1, padding='same')(conv_dropout)
    max_pool_1D_1 = MaxPooling1D(pool_size=max_pool_size, strides=1, padding='same')(conv_dropout)

    conv_layer2 = Conv1D(conv2_filters, window_size, padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer2)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(conv2d_dropout)(conv2D_act)
    # ave_pool_2 = AveragePooling1D(2, 1, padding='same')(conv_dropout)
    max_pool_1D_2 = MaxPooling1D(pool_size=max_pool_size, strides=1, padding='same')(conv_dropout)

    conv_layer3 = Conv1D(conv3_filters, window_size,kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer3)
    conv2D_act = activations.relu(batch_norm)
    conv_dropout = Dropout(conv2d_dropout)(conv2D_act)
    max_pool_1D_3 = MaxPooling1D(pool_size=max_pool_size, strides=1, padding='same')(conv_dropout)
    # ave_pool_3 = AveragePooling1D(2, 1, padding='same')(conv_dropout)

    #concat pooling layers
    conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])

    ######## Recurrent Layers ########
    if (recurrent_layer == 'lstm'):
        if (bidirection):
            print('Entering LSTM Layers')
            #Creating Bidirectional LSTM layers
            # lstm_f1 = Bidirectional(LSTM(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout, recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer))(conv_features)
            lstm_f1 = Bidirectional(LSTM(400,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(conv_features)
            lstm_f2 = Bidirectional(LSTM(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5))(lstm_f1)
            # lstm_f2 = Bidirectional(LSTM(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer))(lstm_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, conv_features])
            concat_features = Dropout(after_recurrent_dropout)(concat_features)
            print('Concatenated LSTM layers')

        else:
            #Creating unidirectional LSTM Layers
            lstm_f1 = LSTM(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer)(conv_features)

            lstm_f2 = LSTM(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer)(lstm_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, conv_features])
            concat_features = Dropout(after_recurrent_dropout)(concat_features)


    elif (recurrent_layer == 'gru'):
        if (bidirection):

            #Creating Bidirectional GRU layers
            gru_f1 = Bidirectional(GRU(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer))(conv_features)

            gru_f2 = Bidirectional(GRU(recurrent_layer2, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer))(gru_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv_features])
            concat_features = Dropout(after_recurrent_dropout)(concat_features)


        else:
            #Creating unidirectional GRU Layers
            gru_f1 = GRU(recurrent_layer1,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer)(conv_features)

            gru_f2 = GRU(recurrent_layer1, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=recurrent_dropout,recurrent_dropout=recurrent_recurrent_dropout, kernel_initializer=recurrent_initalizer)(gru_f1)

            #concatenate LSTM with convolutional layers
            concat_features = Concatenate(axis=-1)([gru_f1, gru_f2, conv_features])
            concat_features = Dropout(after_recurrent_dropout)(concat_features)
    else:

        print('Only LSTM and GRU recurrent layers are used in this model')
        return

    #Dense Fully-Connected DNN layers
    # concat_features = Flatten()(concat_features)
    fc_dense1 = Dense(dense_1, activation='relu', kernel_initializer=dense_initializer)(concat_features)
    # fc_dense1 = Dense(dense_1, activation='relu', kernel_initializer=dense_initializer)(conv_features)
    fc_dense1_dropout = Dropout(dense_dropout)(fc_dense1)
    fc_dense2 = Dense(dense_2, activation='relu', kernel_initializer=dense_initializer)(fc_dense1_dropout)
    fc_dense2_dropout = Dropout(dense_dropout)(fc_dense2)
    fc_dense3 = Dense(dense_3, activation='relu', kernel_initializer=dense_initializer)(fc_dense2_dropout)
    fc_dense3_dropout = Dropout(dense_dropout)(fc_dense3)
    fc_dense4 = Dense(dense_4, activation='relu', kernel_initializer=dense_initializer)(fc_dense3_dropout)
    fc_dense4_dropout = Dropout(dense_dropout)(fc_dense4)


    #Final Output layer with 8 nodes for the 8 output classifications
    # main_output = Dense(8, activation='softmax', name='main_output')(concat_features)
    main_output = Dense(8, activation='softmax', name='main_output')(fc_dense4_dropout)

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
    logs_path = str(args.logs_dir)

    all_data = 0.01
    epochs = 1
    # logs_path = job_dir + 'logs/tensorboard/' + 'hp_config_'+str(datetime.date(datetime.now()))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    print('TensorBoard logs stored at: ', logs_path)

    #if all_data argument not b/w 0 and 1 then its set to default value - 0.5
    if (all_data == 0 or all_data > 1):
        all_data = 0.5

    #Load data
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)
    test_hot, testpssm, testlabel = load_cb513(all_data)

    model = build_model_hpconfig(args)

    print('Fitting model...')
    # with tf.device('/device:GPU:0'): - if using GPU
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],shuffle=True)

    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,
            batch_size=1,callbacks=[tensorboard])
    #evaluate with casp10 and casp11 test datasets
    eval_score = score[1]

    print('Model Loss : ', score[0])
    print('Model Accuracy : %.2f%% ', (eval_score*100))
    print('Model Accuracy : ', score[1])

    #evaluation storage metadata

    # #Initialise Hypertuning
    # hpt = hypertune.HyperTune()
    # hpt.report_hyperparameter_tuning_metric(
    #     hyperparameter_metric_tag='eval_score',
    #     metric_value=eval_score,
    #     global_step=1000
    # )

    model_save_path = 'model_hptuning_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '_accuracy-'+ str(score[1]) \
        +'_loss-' + str(score[0]) + '.h5'

    # upload_history(history,model_save_path,score)
    # upload_model(model, args,model_save_path)
    # plot_history(history.history, show_histograms=True, show_boxplots=True, show_kde=True)


parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=42,
                    help='batch size for training data (default: 42)')
# parser.add_argument('--result_dir', type=str, default='./result',
#                     help='Output directory (default: ./result)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('-recurrent_layer1', '--recurrent_layer1', type=int, default=400,
                    help ='The number of nodes for first recurrent hidden layer')

parser.add_argument('-recurrent_layer2', '--recurrent_layer2', type=int, default=300,
                    help ='The number of nodes for second recurrent hidden layer')

parser.add_argument('-recurrent_dropout', '--recurrent_dropout', default=0.5,
                    help ='Dropout for recurrent hidden Layers')

parser.add_argument('-recurrent_recurrent_dropout', '--recurrent_recurrent_dropout', default=0.5,
                    help ='Recurrent Dropout for recurrent hidden Layers')

parser.add_argument('-recurrent_weight_initializer', '--recurrent_weight_initializer',
                    help='Weight Initializer for Recurrent Layers',required=False, default = 'glorot_uniform')

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
parser.add_argument('-alldata', '--alldata', type =float, default=0.50,
                    help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc, Default: 0.50')
parser.add_argument('-bidirection', '--bidirection', type =bool, default=True,
                    help='Select whether you want the LSTM to be unidirectional or bidirectional - True = BDLSTM, False - ULSTM, default: True')
parser.add_argument('-batch_norm', '--batch_norm', type =bool, default=True,
                    help='Select whether you want a BatchNormalization layer b/w and after Convolutional layers default: True')
parser.add_argument('-recurrent_layer', '--recurrent_layer',  type = str.lower, default='lstm',
                    help='Select what recurrent layer to use in network - GRU or LSTM, default: LSTM')
parser.add_argument('-lr', '--learning_rate',
                    help='Learning rate for training model',required=False, default = .003)
parser.add_argument('-conv1_filters', '--conv1_filters',
                    help='conv1_filters',required=False, default = 64)
parser.add_argument('-conv2_filters', '--conv2_filters',
                    help='conv2_filters',required=False, default = 128)
parser.add_argument('-conv3_filters', '--conv3_filters',
                    help='conv3_filters',required=False, default = 256)
parser.add_argument('-conv_weight_initializer', '--conv_weight_initializer',
                    help='Weight Initializer for Conv Layers',required=False, default = 'glorot_uniform')
parser.add_argument('-window_size', '--window_size',
                    help='window_size',required=False, default = 7)
parser.add_argument('-kernel_regularizer', '--kernel_regularizer',
                    help='kernel_regularizer',required=False, default = "l2")
parser.add_argument('-conv2d_dropout', '--conv2d_dropout', help='dropout',required=False, default = 0.5)

parser.add_argument('-pool_size', '--pool_size',
                    help='pool_size',required=False, default = 2)
parser.add_argument('-logs_dir', '--logs_dir',
                    help='Directory on cloud storage for Tensorboard logs',required=False, default = (BUCKET_NAME + "/logs/tensorboard"))
parser.add_argument('-dense_dropout', '--dense_dropout',
                    help='dense_dropout',required=False, default = 0.5)
parser.add_argument('-dense_1', '--dense_1',
                    help='dense_dropout',required=False, default = 300)
parser.add_argument('-dense_2', '--dense_2',
                    help='dense_dropout',required=False, default = 100)
parser.add_argument('-dense_3', '--dense_3',
                    help='dense_dropout',required=False, default = 50)
parser.add_argument('-dense_4', '--dense_4',
                    help='dense_dropout',required=False, default = 16)
parser.add_argument('-dense_weight_initializer', '--dense_weight_initializer',
                    help='Weight Initializer for Dense Layers',required=False, default = 'glorot_uniform')


args = parser.parse_args()
print(args.epochs)

main(args)
