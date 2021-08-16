#########################################################################
###     Dummy model - Used for training and test purposes Only        ###
#########################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Embedding, Dense, Dropout, Activation, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model(params):

    """
    Description:
        Building DCBLSTM model
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

    #get shape of input layers
    print ("Protein Sequence shape: ", main_input.get_shape())
    print ("Protein Profile shape: ",auxiliary_input.get_shape())

    #concatenate 2 input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    ######## 3x1D-Convolutional Layers with BatchNormalization, Dropout and MaxPooling ########

    conv_layer1 = Conv1D(filters=params["model_parameters"][0]["conv_layer1_filters"], kernel_size=params["model_parameters"][0]["conv_layer1_window"],
        kernel_regularizer = params["model_parameters"][0]["conv_layer_kernel_regularizer"], padding=params["model_parameters"][0]["conv_layer_padding"],
            strides=params["model_parameters"][0]["conv_layer_stride"], activation=params["model_parameters"][0]["conv_layer_activation"],
                kernel_initializer=params["model_parameters"][0]["conv_layer_kernel_initializer"])(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv1_dropout = Dropout(params["model_parameters"][0]["conv_layer1_dropout"])(batch_norm)

    #Dense Fully-Connected DNN layers
    after_lstm_dense1 = Dense(params["model_parameters"][0]["dense_layer1_units"], activation=params["model_parameters"][0]["dense_activation"])(conv1_dropout)
    after_lstm_dense1_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_lstm_dense1)

    after_lstm_dense2 = Dense(params["model_parameters"][0]["dense_layer2_units"], activation=params["model_parameters"][0]["dense_activation"])(after_lstm_dense1_dropout)
    after_lstm_dense2_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_lstm_dense2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(params["model_parameters"][0]["dense_layer3_units"], activation=params["model_parameters"][0]["dense_classification"],
        name='main_output')(after_lstm_dense2_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    if params["parameters"][0]["optimizer"].lower() == "sgd":
        optimizer = SGD(learning_rate=params["parameters"][0]["learning_rate"])
    if params["parameters"][0]["optimizer"].lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=params["parameters"][0]["learning_rate"])
    if params["parameters"][0]["optimizer"].lower() == "adadelta":
        optimizer = Adadelta(learning_rate=params["parameters"][0]["learning_rate"])
    if params["parameters"][0]["optimizer"].lower() == "adagrad":
        optimizer = Adagrad(learning_rate=params["parameters"][0]["learning_rate"])
    else:
        optimizer = Adam(learning_rate=params["parameters"][0]["learning_rate"])

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer=optimizer, loss={'main_output': 'categorical_crossentropy'},
        metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(),
            TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision(), AUC()])

    #print model summary
    model.summary()

    return model
