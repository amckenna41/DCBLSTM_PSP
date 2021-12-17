################################################################################
######     Dummy model - Used for training and test purposes only         ######
################################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Embedding, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
from tensorflow.keras.metrics import AUC, MeanSquaredError, RootMeanSquaredError,
    FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall

def build_model(params):
    """
    Description:
        Building small dummy neural network model to test code pipeline is
        working. Due to the computational complexity of the main deep neural networks
        used, a simpler and faster version of them was built. Dummy model consists
        of a 1 layer Convolutional neural network, 2 fully-connected layers and a
        final 8-node output layer, and takes the same protein training data as input.
    Args:
        :params (dict): dictionary of various model and GCP parameters to use in
        building and training of network.
    Returns:
        :model (keras.model): trained Tensorflow Keras ML model.
    """
    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(params["input"]["input_shape"],), dtype=params["input"]["dtype"], name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=params["input"]["num_aminoacids"], input_dim=params["input"]["num_aminoacids"], \
        input_length=params["input"]["input_shape"], name="embedding")(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(params["input"]["input_shape"],params["input"]["num_aminoacids"]), name='aux_input')

    #concatenate 2 input layers
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    ######## 1x1D-Convolutional Layers with Dropout ########
    conv_layer1 = Conv1D(**{**params["conv"], **params["conv1"]})(concat)
    conv1_dropout = Dropout(**params["dropout1"])(conv_layer1)

    ########  Dense Fully-Connected DNN layers  ########
    conv_dense = Dense(**params["dense1"])(conv1_dropout)
    conv_dropout = Dropout(**params["dropout2"])(conv_dense)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(**params["dense2"])(conv_dropout)

    #create model object from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #Optimizers are algorithms or methods used to change the attributes of your
    #neural network such as weights and learning rate in order to reduce the losses.
    #Adam optimizer used by default
    if (params["optimizer"]["name"].lower() == "sgd"):
        optimizer = SGD(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "rmsprop"):
        optimizer = RMSprop(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "adadelta"):
        optimizer = Adadelta(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "adagrad"):
        optimizer = Adagrad(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "adamax"):
        optimizer = Adamax(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "nadam"):
        optimizer = Nadam(**params["optimizer"], name="SGD")
    elif (params["optimizer"]["name"].lower() == "ftrl"):
        optimizer = Ftrl(**params["optimizer"], name="SGD")
    else:
        optimizer = Adam(**params["optimizer"])

    #compile model using optimizer and the cateogorical crossentropy loss function,
    #including all the metrics to be captured during training process
    model.compile(optimizer=optimizer, loss={'main_output': 'categorical_crossentropy'},
        metrics=['accuracy', MeanSquaredError(), RootMeanSquaredError(), FalseNegatives(), FalsePositives(),
            TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision(), AUC()])

    model._name = "dummy_model"

    #print model summary
    model.summary()

    return model
