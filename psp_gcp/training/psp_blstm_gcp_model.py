#PSP model using BLSTM RNN with CNN + DNN

#import required modules and dependancies
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Input, Conv1D, Embedding, LSTM, Dense, Dropout, Activation, Convolution2D, GRU, Concatenate, Reshape,MaxPooling1D, Conv2D, MaxPooling2D,Convolution1D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
# from tensorflow.keras.utils import plot_model
import pandas as pd
# from io import BytesIO
# from tensorflow.python.lib.io import file_io
import os
import sys
from datetime import date
from datetime import datetime
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
current_datetime = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

#building BLSTM_3xConv_Model
def build_model():

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(700,), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=21, input_dim=21, input_length=700, name="embedding")(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(700,21), name='aux_input')
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)

    #get shape of input layers
    print ("Protein Sequence shape: ", main_input.get_shape())
    print ("Protein Profile shape: ",auxiliary_input.get_shape())

    #concatenate 2 input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    #3x1D Convolutional Hidden Layers with BatchNormalization, Dropout and MaxPooling
    conv_layer1 = Conv1D(16, 7, padding='same', name ="Conv1D_1")(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    # ave_pool_1 = AveragePooling1D(2, 1, padding='same')(conv_dropout)
    max_pool_1D_1 = MaxPooling1D(pool_size=2, strides=1, padding='same', name="MaxPool_1")(conv_dropout)

    conv_layer2 = Conv1D(32, 7, padding='same', name="Conv1D_2")(concat)
    batch_norm = BatchNormalization()(conv_layer2)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    max_pool_1D_2 = MaxPooling1D(pool_size=2, strides=1, padding='same',  name="MaxPool_2")(conv_dropout)

    conv_layer3 = Conv1D(64, 7, padding='same', name="Conv1D_3")(concat)
    batch_norm = BatchNormalization()(conv_layer3)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    max_pool_1D_3 = MaxPooling1D(pool_size=2, strides=1, padding='same',  name="MaxPool_3")(conv_dropout)

    #concatenate convolutional layers
    conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])
    print("Shape of convolutional output: ", conv_features.get_shape())

    lstm_dense = Dense(600, activation='relu', name="after_cnn_dense")(conv_features)

     ######## Recurrent Bi-Directional Long-Short-Term-Memory Layers ########
    lstm_f1 = Bidirectional(LSTM(200,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5, name="BLSTM_1"))(lstm_dense)

    lstm_f2 = Bidirectional(LSTM(200, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5, name="BLSTM_2"))(lstm_f1)

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=2)([lstm_f1, lstm_f2, lstm_dense])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    dense_1 = Dense(600, activation='relu', name="after_rnn_dense")(concat_features)
    dense_1_dropout = Dropout(0.3)(dense_1)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = (Dense(8, activation='softmax', name='main_output'))(dense_1_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.0015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])
    model.summary()

    #set earlyStopping and checkpoint callback
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = "checkpoints/CDBLSTM_" + current_datetime + ".h5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')

    return model

#main function to train and evaluate CNN + RNN + DNN model
def main(args):

    #setting parsed input arguments
    job_dir = str(args.job_dir)
    all_data = float(args.alldata)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)

    print("Logs Path: ", logs_path)
    print('Job Logs: ', job_dir)
    # 
    # epochs = 1
    # all_data = 0.1
    # batch_size = 150
    #if all_data argument not b/w 0 and 1 then its set to default value - 0.5
    if (all_data == 0 or all_data > 1):
        all_data = 0.5

    print('Running model using {}%% of data'.format(int(all_data*100)))
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)
    test_hot, testpssm, testlabel = load_cb513(all_data)


    #build model
    print('Building 3x1Dconv BLSTM model')
    model = build_model()

    # model_plot_name = 'model_plot' + current_datetime + '.png'
    # plot_model(model, to_file=model_plot_name, show_shapes=True, show_layer_names=True)
    #https://github.com/XifengGuo/CapsNet-Keras/issues/25

    #initialise model callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath="blstm_3conv_checkpoint/", verbose=1,save_best_only=True, monitor='val_acc', mode='max')

    # with tf.device('/gpu:0'): #use for training with GPU on TF
    print('Fitting model...')
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard, checkpoint],shuffle=True)


    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,batch_size=1)
    # eval_score = score[1]

    #initialise TensorBoard summary variables
    loss_summary = tf.summary.scalar(name='Loss Summary', data=score[0])
    accuracy_summary = tf.summary.scalar(name='Accuracy Summary', data=score[1])

    print("Training Accuracy: ", max(history.history['accuracy']))
    print("Training Loss: ", min(history.history['loss']))

    print('Evaluation Loss : ', score[0])
    print('Evaluation Accuracy : ', score[1])

    model_blob_path = 'models/model_blstm_3x1Dconv_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + current_datetime

    model_save_path = 'model_blstm_3x1Dconv_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + current_datetime + \
        '_accuracy-'+ str(score[1]) +'_loss-' + str(score[0]) + '.h5'

    #create directory in bucket for new model - name it the model name, store model
    # upload_history(history,score,model_save_path)
    # upload_model(model, model_blob_path, model_save_path)
    # plot_history(history.history, model_blob_path,show_histograms=False, show_boxplots=False, show_kde=False)

#initialise input arguments to model
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=120,
                    help='batch size for training data (default: 120)')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')

parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = BUCKET_PATH)

parser.add_argument('-alldata', '--alldata', type =float, default=1,
                    help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc')

parser.add_argument('-logs_dir', '--logs_dir', help='Directory on cloud storage for Tensorboard logs',required=False, default = (BUCKET_NAME + "/logs/tensorboard"))
args = parser.parse_args()


main(args)
