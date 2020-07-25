import os
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
from tensorflow import keras
import argparse
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, Concatenate, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import random as rn
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io
#
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
#
# np.random.seed(2018)
# rn.seed(2018)

def load_cul6133_filted():
    '''
    TRAIN data Cullpdb+profile_6133_filtered
    Test data  CB513 CASP10 CASP11
    '''
    print("Loading train data (Cullpdb_filted)...")

    f = BytesIO(file_io.read_file_to_string('gs://keras-python-models/cullpdb+profile_6133_filtered.npy', binary_mode=True))
    data = np.load(f)
    #data = np.load('cullpdb+profile_6133_filtered.npy')
    data = np.reshape(data, (-1, 700, 57))
    # print data.shape
    datahot = data[:, :, 0:21]#sequence feature
    # print 'sequence feature',dataonehot[1,:3,:]
    datapssm = data[:, :, 35:56]#profile feature
    # print 'profile feature',datapssm[1,:3,:]
    labels = data[:, :, 22:30]    # secondary struture label , 8-d
    np.random.seed(2018)
    # shuffle data
    num_seqs, seqlen, feature_dim = np.shape(data)
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)#
    np.random.shuffle(seq_index)

    # #train data
    # trainhot = datahot[seq_index[:5278]] #21
    # trainlabel = labels[seq_index[:5278]] #8
    # trainpssm = datapssm[seq_index[:5278]] #21

    trainhot = datahot[seq_index[:1500]] #21
    trainlabel = labels[seq_index[:1500]] #8
    trainpssm = datapssm[seq_index[:1500]]


    #val data
    vallabel = labels[seq_index[1500:1575]] #8
    valpssm = datapssm[seq_index[1500:1575]] # 21
    valhot = datahot[seq_index[1500:1575]] #21


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
    # design the deepaclstm model
    main_input = Input(shape=(700,), dtype='float32', name='main_input')
    #main_input = Masking(mask_value=23)(main_input)
    x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
    auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)
    print (main_input.get_shape())
    print (auxiliary_input.get_shape())
    #concat = merge([x, auxiliary_input], mode='concat', concat_axis=-1)
    concat = Concatenate(axis=-1)([x, auxiliary_input])

    conv1_features = Convolution1D(42,1,activation='relu', padding='same')(concat)
    # print 'conv1_features shape', conv1_features.get_shape()
    conv1_features = Reshape((700, 42, 1))(conv1_features)

    conv2_features = Convolution2D(42,3,1,activation='relu', padding='same')(conv1_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()

    conv2_features = Reshape((700,42*42))(conv2_features)
    conv2_features = Dropout(0.5)(conv2_features)
    conv2_features = Dense(400, activation='relu')(conv2_features)

    #, activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5
    lstm_f1 = LSTM(300,return_sequences=True,activation = 'tanh', recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)
    lstm_b1 = LSTM(300, return_sequences=True, activation='tanh',go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(conv2_features)

    lstm_f2 = LSTM(300, return_sequences=True,activation = 'tanh',recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_f1)
    lstm_b2 = LSTM(300, return_sequences=True,activation='tanh', go_backwards=True,recurrent_activation='sigmoid',dropout=0.5,recurrent_dropout=0.5)(lstm_b1)

    #concat_features = merge([lstm_f2, lstm_b2, conv2_features], mode='concat', concat_axis=-1)
    concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])
    concat_features = Dropout(0.4)(concat_features)
    protein_features = Dense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)

    main_output = Dense(8, activation='softmax', name='main_output')(protein_features)


    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    adam = Adam(lr=0.003)
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    model.summary()

    return model

def main(job_dir, **args):
    logs_path = job_dir + 'logs/tensorboard'

    with tf.device('/device:GPU:0'):
        #Load data
        train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted()

        #build model
        model = build_model()

        #add callbacks for tesnorboard and history
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        #fit model
        model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=10, batch_size=42, verbose=2, callbacks=[tensorboard],shuffle=True)

        #save model
        model.save('model_1.h5')
        with file_io.FileIO('model_1.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model_1.h5', mode='w+') as output_f:
                output_f.write(input_f.read())

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
