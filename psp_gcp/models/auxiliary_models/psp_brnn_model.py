
################################################################################
######                     BRNN - Bidirectional RNN                       ######
################################################################################

#import required modules and dependancies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Flatten, Input, Conv1D, Embedding, Dense, Dropout, Activation,  Concatenate, Reshape,MaxPooling1D, BatchNormalization,ReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model(params):

    """
    Description:
        Building BRNN model
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


    ########  Bi-Directional Recurrent network ########
    brnn1 = Bidirectional(SimpleRNN(params["model_parameters"][0]["rnn_layer1_units"],return_sequences=True,
        activation = params["model_parameters"][0]["rnn_layer_activation"], recurrent_activation=params["model_parameters"][0]["rnn_recurrent_activation"],
            dropout=params["model_parameters"][0]["rnn_layer1_dropout"],recurrent_dropout=params["model_parameters"][0]["rnn_layer1_recurrent_dropout"]))(conv3_dropout)

    brnn2 = Bidirectional(SimpleRNN(params["model_parameters"][0]["rnn_layer2_units"],return_sequences=True,
        activation = params["model_parameters"][0]["rnn_layer_activation"], recurrent_activation=params["model_parameters"][0]["rnn_recurrent_activation"],
            dropout=params["model_parameters"][0]["rnn_layer2_dropout"],recurrent_dropout=params["model_parameters"][0]["rnn_layer2_recurrent_dropout"]))(brnn1)

    ############################################################################################

    #concatenate simple RNN with convolutional layers
    concat_features = Concatenate(axis=-1)([brnn1, brnn2, conv3_dropout])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    after_rnn_dense1 = Dense(params["model_parameters"][0]["dense_layer1_units"], activation=params["model_parameters"][0]["dense_activation"])(concat_features)
    after_rnn_dense1_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_rnn_dense1)

    after_rnn_dense2 = Dense(params["model_parameters"][0]["dense_layer2_units"], activation=params["model_parameters"][0]["dense_activation"])(concat_features)
    after_rnn_dense2_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_rnn_dense2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(params["model_parameters"][0]["dense_layer3_units"], activation=params["model_parameters"][0]["dense_classification"],
        name='main_output')(after_rnn_dense2_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    if params["parameters"][0]["optimizer"].lower() == "sgd":
        pass
    if params["parameters"][0]["optimizer"].lower() == "rmsprop":
        pass
    if params["parameters"][0]["optimizer"].lower() == "adadelta":
        pass
    if params["parameters"][0]["optimizer"].lower() == "adagrad":
        pass
    else:
        optimizer = Adam(lr=params["parameters"][0]["learning_rate"])

    #compile model using adam optimizer and the cateogorical crossentropy loss function
    model.compile(optimizer=optimizer, loss={'main_output': 'categorical_crossentropy'},
        metrics=['accuracy', MeanSquaredError(), FalseNegatives(), FalsePositives(),
            TrueNegatives(), TruePositives(), MeanAbsoluteError(), Recall(), Precision(), AUC()])

    print('building this model here')
    #print model summary
    model.summary()

    return model
