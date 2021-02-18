#################################################
### PSP_CD - PSP - Convolutional Deep Network ###
#################################################

#import required modules and dependancies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Embedding, Dense, Dropout, Activation, Concatenate, Reshape,MaxPooling1D, BatchNormalization
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
    concat = Concatenate(axis=-1)([embed, auxiliary_input])

    ######## 3x1D-Convolutional Layers with BatchNormalization, Dropout and MaxPooling ########

    conv_layer1 = Conv1D(16, 7, kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer1)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    max_pool_1D_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    conv_layer2 = Conv1D(32, 7, padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer2)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    max_pool_1D_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    conv_layer3 = Conv1D(64, 7,kernel_regularizer = "l2", padding='same')(concat)
    batch_norm = BatchNormalization()(conv_layer3)
    conv_act = activations.relu(batch_norm)
    conv_dropout = Dropout(0.2)(conv_act)
    max_pool_1D_3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_dropout)

    ############################################################################################

    #concatenate convolutional layers
    conv_features = Concatenate(axis=-1)([max_pool_1D_1, max_pool_1D_2, max_pool_1D_3])
    concat_features = Dropout(0.4)(conv_features)

    #Dense Fully-Connected DNN layers
    after_cnn_dense = Dense(600, activation='relu')(concat_features)
    after_cnn_dense_dropout = Dropout(0.3)(after_cnn_dense)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(8, activation='softmax', name='main_output')(after_cnn_dense_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    adam = Adam(lr=0.00015)

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision()])

    #print model summary
    model.summary()

    return model
