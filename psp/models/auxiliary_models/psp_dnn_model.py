################################################################################
#######                  PSP-DNN - Deep Neural Network                  ########
################################################################################

#import required modules and dependancies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Embedding, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, RootMeanSquaredError,
    FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall

def build_model(params):
    """
    Description:
        Building PSP-DNN neural network model. A network consisting of 7 fully-connected layers.
    Args:
        :params (dict): dictionary of various model parameters to use in
        building and training of network.
    Returns:
        :model (keras.model): trained Tensorflow Keras ML model.
    """
    #main input is the length of the amino acid in the protein sequence (700,)
    main_input = Input(shape=(params["input"]["input_shape"],), dtype=params["input"]["dtype"], name='main_input')

    #Embedding Layer used as input to the neural network
    embed = Embedding(output_dim=params["input"]["num_aminoacids"], input_dim=params["input"]["num_aminoacids"], \
        input_length=params["input"]["input_shape"])(main_input)

    #secondary input is the protein profile features
    auxiliary_input = Input(shape=(params["input"]["input_shape"],params["input"]["num_aminoacids"]), name='aux_input')

    #concatenate 2 input layers
    concat_features = Concatenate(axis=-1)([embed, auxiliary_input])

    #############################################################################

    #Dense Fully-Connected DNN layers
    dense_1 = Dense(**params["dense1"])(concat_features)
    dense_1_dropout = Dropout(**params["dropout1"])(dense_1)
    dense_2 = Dense(**params["dense2"])(dense_1_dropout)
    dense_2_dropout = Dropout(**params["dropout2"])(dense_2)
    dense_3 = Dense(**params["dense3"])(dense_2_dropout)
    dense_3_dropout = Dropout(**params["dropout3"])(dense_3)
    dense_4 = Dense(**params["dense4"])(dense_3_dropout)
    dense_4_dropout = Dropout(**params["dropout4"])(dense_4)
    dense_5 = Dense(**params["dense5"])(dense_4_dropout)
    dense_5_dropout = Dropout(**params["dropout5"])(dense_5)
    dense_6 = Dense(**params["dense6"])(dense_5_dropout)
    dense_6_dropout = Dropout(**params["dropout6"])(dense_6)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(**params["dense7"])(dense_6_dropout)

    #create model from inputs and outputs
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

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = optimizer, loss={'main_output': 'categorical_crossentropy'}, \
        metrics=['accuracy', MeanSquaredError(), RootMeanSquaredError(), FalseNegatives(),
            FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    model._name = "psp_dnn_model"

    #print model summary
    model.summary()

    return model
