########################################################################################
###     DCBRNN - Deep Convolutional Bidirectional Simple Recurrent Neural network    ###
########################################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Input, Conv1D, Embedding, Dense, Dropout, Activation, Concatenate, Reshape,MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model(params):

    """
    Description:
        Building DCRNN model
    Args:
        None
    Returns:
        None

    """

    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(params["model_parameters"][0]["input_shape"],), dtype='float32', name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=params["model_parameters"][0]["num_aminoacids"], input_dim=params["model_parameters"][0]["num_aminoacids"], input_length=params["model_parameters"][0]["input_shape"])(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(params["model_parameters"][0]["input_shape"],params["model_parameters"][0]["num_aminoacids"]), name='aux_input')

    #concatenate 2 input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    ######## 3x1D-Convolutional Layers with BatchNormalization, Dropout and MaxPooling ########

    conv_layer1 = Conv1D(filters=params["model_parameters"][0]["conv_layer1_filters"], kernel_size=params["model_parameters"][0]["conv_layer1_window"],
        kernel_regularizer = params["model_parameters"][0]["conv_layer_kernel_regularizer"], padding=params["model_parameters"][0]["conv_layer_padding"],
            strides=params["model_parameters"][0]["conv_layer_stride"], activation=params["model_parameters"][0]["conv_layer_activation"],
                kernel_initializer=params["model_parameters"][0]["conv_layer_kernel_initializer"])(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv1_dropout = Dropout(params["model_parameters"][0]["conv_layer1_dropout"])(batch_norm)

    conv_layer2 = Conv1D(filters=params["model_parameters"][0]["conv_layer2_filters"], kernel_size=params["model_parameters"][0]["conv_layer2_window"],
        kernel_regularizer = params["model_parameters"][0]["conv_layer_kernel_regularizer"], padding=params["model_parameters"][0]["conv_layer_padding"],
            strides=params["model_parameters"][0]["conv_layer_stride"], activation=params["model_parameters"][0]["conv_layer_activation"],
                kernel_initializer=params["model_parameters"][0]["conv_layer_kernel_initializer"])(conv1_dropout)
    batch_norm = BatchNormalization()(conv_layer2)
    conv2_dropout = Dropout(params["model_parameters"][0]["conv_layer2_dropout"])(batch_norm)

    conv_layer3 = Conv1D(filters=params["model_parameters"][0]["conv_layer3_filters"], kernel_size=params["model_parameters"][0]["conv_layer3_window"],
        kernel_regularizer = params["model_parameters"][0]["conv_layer_kernel_regularizer"], padding=params["model_parameters"][0]["conv_layer_padding"],
            strides=params["model_parameters"][0]["conv_layer_stride"], activation=params["model_parameters"][0]["conv_layer_activation"],
                kernel_initializer=params["model_parameters"][0]["conv_layer_kernel_initializer"])(conv2_dropout)
    batch_norm = BatchNormalization()(conv_layer3)
    conv3_dropout = Dropout(params["model_parameters"][0]["conv_layer3_dropout"])(batch_norm)

    ############################################################################################

    #concatenate convolutional layers
    # conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])

    #dense layer before simple BRNN's
    brnn_dense = Dense(600, activation='relu', name="after_cnn_dense")(conv_features)

    ######## Simple RNN Layers ########
    brnn_f1 = Bidirectional(SimpleRNN(600,return_sequences=True,activation='relu',recurrent_activation='sigmoid',dropout=0.5, recurrent_dropout=0.5, name="simple_rnn_1"))(brnn_dense)

    brnn_f2 = Bidirectional(SimpleRNN(600,return_sequences=True, activation='relu',recurrent_activation='sigmoid',dropout=0.5, recurrent_dropout=0.5, name="simple_rnn_2"))(brnn_f1)

    ############################################################################################

    #concatenate simple BRNN's with convolutional layers
    concat_features = Concatenate(axis=-1)([brnn_f1, brnn_f2, brnn_dense])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    after_brnn_dense = Dense(600, activation='relu')(concat_features)
    after_brnn_dense_dropout = Dropout(0.3)(after_brnn_dense)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(after_brnn_dense_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.00015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    #print model summary
    model.summary()

    return model
