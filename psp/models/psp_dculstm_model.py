#########################################################################
### DCULSTM - Deep Convolutional Unidirectional Long short-term memory ###
#########################################################################

#import required modules and dependancies
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Unidirectional, LSTM, Input, Conv1D, Embedding, Dense, Dropout, Activation,  Concatenate, Reshape,MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations

def build_model(params):
    """
    Description:
        Building DCULSTM neural network model.
    Args:
        params (dict): dictionary of various model and GCP parameters to use in
        building and training of network.
    Returns:
        model (keras.model): trained Tensorflow Keras ML model.
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
    # conv_features = Concatenate(axis=-1)([conv1_dropout, conv2_dropout, conv3_dropout])

    #dense layer before LSTM's
    # lstm_dense = Dense(600, activation='relu', name="after_cnn_dense")(conv_features)

    ######## Recurrent Bi-Directional Long-Short-Term-Memory Layers ########
    lstm_f1 = LSTM(params["model_parameters"][0]["lstm_layer1_units"],return_sequences=True,
        activation = params["model_parameters"][0]["lstm_layer_activation"], recurrent_activation=params["model_parameters"][0]["lstm_recurrent_activation"],
            dropout=params["model_parameters"][0]["lstm_layer1_dropout"],recurrent_dropout=params["model_parameters"][0]["lstm_layer1_recurrent_dropout"])(conv3_dropout)

    lstm_f2 = LSTM(params["model_parameters"][0]["lstm_layer2_units"],return_sequences=True,
        activation = params["model_parameters"][0]["lstm_layer_activation"], recurrent_activation=params["model_parameters"][0]["lstm_recurrent_activation"],
            dropout=params["model_parameters"][0]["lstm_layer2_dropout"],recurrent_dropout=params["model_parameters"][0]["lstm_layer2_recurrent_dropout"])(lstm_f1)

    lstm_f3 = LSTM(params["model_parameters"][0]["lstm_layer3_units"],return_sequences=True,
        activation = params["model_parameters"][0]["lstm_layer_activation"], recurrent_activation=params["model_parameters"][0]["lstm_recurrent_activation"],
            dropout=params["model_parameters"][0]["lstm_layer3_dropout"],recurrent_dropout=params["model_parameters"][0]["lstm_layer3_recurrent_dropout"])(lstm_f2)

    ############################################################################################

    #concatenate LSTM with convolutional layers
    concat_features = Concatenate(axis=-1)([lstm_f1, lstm_f2, lstm_f3, conv3_dropout])
    concat_features = Dropout(0.4)(concat_features)

    #Dense Fully-Connected DNN layers
    after_lstm_dense1 = Dense(params["model_parameters"][0]["dense_layer1_units"], activation=params["model_parameters"][0]["dense_activation"])(concat_features)
    after_lstm_dense1_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_lstm_dense1)

    after_lstm_dense2 = Dense(params["model_parameters"][0]["dense_layer2_units"], activation=params["model_parameters"][0]["dense_activation"])(concat_features)
    after_lstm_dense2_dropout = Dropout(params["model_parameters"][0]["dense_dropout"])(after_lstm_dense2)

    #Final Dense layer with 8 nodes for the 8 output classifications
    main_output = Dense(params["model_parameters"][0]["dense_layer3_units"], activation=params["model_parameters"][0]["dense_classification"],
        name='main_output')(after_lstm_dense2_dropout)

    #create model from inputs and outputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])

    #use Adam optimizer
    if params["parameters"][0]["optimizer"].lower() == "sgd":
        optimizer = SGD(learning_rate=params["parameters"][0]["learning_rate"], name="SGD")
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


# **append each jobs results to a the one csv file.
