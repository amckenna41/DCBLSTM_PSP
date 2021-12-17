################################################################################
## DCBRNN - Deep Convolutional Bidirectional Simple Recurrent Neural network  ##
################################################################################

#import required modules and dependancies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Input, Conv1D, Embedding, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, RootMeanSquaredError,
    FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall

def build_model(params):
    """
    Description:
        Building PSP-DCBRNN neural network model. 3x1DConv convolutional layers
        followed by 2 bidirectional simple RNN recurrent layers and 3 dense fully-connected layers.
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
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    ####### 3x1D-Convolutional Layers with BatchNormalization and Dropout ######

    conv_layer1 = Conv1D(**{**params["conv"], **params["conv1"]})(concat)
    batch_norm = BatchNormalization(**params["batch_norm"])(conv_layer1)
    conv1_dropout = Dropout(**params["dropout1"])(batch_norm)

    conv_layer2 = Conv1D(**{**params["conv"], **params["conv2"]})(concat)
    batch_norm = BatchNormalization(**params["batch_norm"])(conv_layer2)
    conv2_dropout = Dropout(**params["dropout2"])(batch_norm)

    conv_layer3 = Conv1D(**{**params["conv"], **params["conv3"]})(concat)
    batch_norm = BatchNormalization(**params["batch_norm"])(conv_layer3)
    conv3_dropout = Dropout(**params["dropout3"])(batch_norm)

    #concatenate convolutional layers
    conv_features = Concatenate(axis=-1)([conv1_dropout, conv2_dropout, conv3_dropout])

    ######## Simple RNN Layers ########
    brnn_f1 = Bidirectional(SimpleRNN(**{**params["brnn"], **params["brnn1"]}))(conv_features)

    brnn_f2 = Bidirectional(SimpleRNN(**{**params["brnn"], **params["brnn2"]}))(brnn_f1)

    #concatenate simple BRNN's with convolutional layers
    concat_features = Concatenate(axis=-1)([brnn_f1, brnn_f2, conv_features])

    ####################  Dense Fully-Connected DNN layers  ####################
    after_brnn_dense = Dense(**params["dense1"])(concat_features)
    after_brnn_dense_dropout = Dropout(**params["dropout4"])(after_brnn_dense)

    #Dense Fully-Connected DNN layers
    after_brnn_dense2 = Dense(**params["dense2"])(concat_features)
    after_brnn_dense2_dropout = Dropout(**params["dropout5"])(after_brnn_dense2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(**params["dense3"])(after_brnn_dense2_dropout)

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

    model._name = "psp_brnn_model"

    #print model summary
    model.summary()

    return model
