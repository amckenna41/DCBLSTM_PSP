# SimpleRNN layer
# Base RNN layer

#########################################################################
###  DCRNN - Deep Convolutional Simple Recurrent Neural network       ###
#########################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Input, Conv1D, Embedding, Dense, Dropout, Activation, Concatenate, Reshape,MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model():

    """
    Description:
        Building DCRNN model
    Args:
        None
    Returns:
        None

    """

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

    ############################################################################################

    #concatenate convolutional layers
    conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])

    #dense layer before simple RNN's
    rnn_dense = Dense(600, activation='relu', name="after_cnn_dense")(conv_features)

    ######## Simple RNN Layers ########
    rnn_f1 = SimpleRNN(600,activation='relu',recurrent_activation='sigmoid',dropout=0.5, recurrent_dropout=0.5, name="simple_rnn_1")(rnn_dense)

    rnn_f2 = SimpleRNN(600,activation='relu',recurrent_activation='sigmoid',dropout=0.5, recurrent_dropout=0.5, name="simple_rnn_2")(rnn_f1)

    ############################################################################################

    #concatenate simple RNN's with convolutional layers
    concat_features = Concatenate(axis=-1)([rnn_f1, rnn_f2, rnn_dense])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    after_rnn_dense = Dense(600, activation='relu')(concat_features)
    after_rnn_dense_dropout = Dropout(0.3)(after_rnn_dense)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(after_rnn_dense_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.00015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    #print model summary
    model.summary()

    return model
