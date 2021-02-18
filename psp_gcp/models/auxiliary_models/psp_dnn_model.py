#################################################
### PSP-DNN - Deep Neural Network ###
#################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Embedding, Dense, Dropout, Activation, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model():

    """
    Description:
        Building PSP-CD model
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
    concat_features = Concatenate(axis=-1)([embed, auxiliary_input])


    ############################################################################################

    #Dense Fully-Connected DNN layers
    dense_1 = Dense(512, activation='relu')(concat_features)
    dense_1_dropout = Dropout(0.3)(dense_1)
    dense_2 = Dense(256, activation='relu')(dense_1_dropout)
    dense_2_dropout = Dropout(0.3)(dense_2)
    dense_3 = Dense(128, activation='relu')(dense_2_dropout)
    dense_3_dropout = Dropout(0.3)(dense_3)
    dense_4 = Dense(64, activation='relu')(dense_3_dropout)
    dense_4_dropout = Dropout(0.3)(dense_4)
    dense_5 = Dense(32, activation='relu')(dense_4_dropout)
    dense_5_dropout = Dropout(0.3)(dense_5)
    dense_6 = Dense(16, activation='relu')(dense_5_dropout)
    dense_6_dropout = Dropout(0.3)(dense_6)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(dense_6_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.00015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    #print model summary
    model.summary()

    return model
