#########################################################################
## Entry script for psp_gcp dir for training on Google Cloud Platform  ##
#########################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Input, Conv1D, Embedding, Dense, Dropout, Activation,  Concatenate, Reshape,MaxPooling1D, Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
import os
from os.path import isfile, join
from os import listdir
import sys
from datetime import date
from datetime import datetime
import importlib
import json
from google.cloud import storage
from json.decoder import JSONDecodeError
from training.training_utils.load_dataset import *
from training.training_utils.plot_model import *
from training.training_utils.gcp_utils import *
from training.training_utils.globals import *
from training.evaluate import *
from models import *
from models.auxiliary_models import *

### Tensorboard parameters and configuration ###
tf.compat.v1.reset_default_graph()
from tensorflow.core.protobuf import rewriter_config_pb2
tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.compat.v1.keras.backend import set_session
config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.gpu_options.allow_growth = True
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.compat.v1.Session(config=config_proto)
set_session(session)

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)

#get model filenames from models directory
all_models = ['psp_dcblstm_model', 'psp_dculstm_model', 'auxiliary_models.psp_cnn_model',\
'auxiliary_models.psp_dcbgru_model','auxiliary_models.psp_dcugru_model','auxiliary_models.psp_dnn_model', \
'auxiliary_models.psp_rbm_model','auxiliary_models.psp_rnn_model','dummy_model']

#main function to train and evaluate CNN + RNN + DNN model
def main(args):

    blob = bucket.blob(args.config_)
    try:
        # Download the contents of the blob as a string and then parse it using json.loads() method
        params = json.loads(blob.download_as_string(client=None))
    except JSONDecodeError as e:
        print('Error getting config JSON file: {}'.format(args.config_))

    #get input arguments
    job_dir = str(args.job_dir)
    training_data = params["parameters"][0]["training_data"]
    filtered = params["parameters"][0]["filtered"]
    batch_size = int(params["parameters"][0]["batch_size"])
    epochs = int(params["parameters"][0]["epochs"])
    logs_path = str(params["parameters"][0]["logs_path"])
    data_dir = str(params["parameters"][0]["data_dir"])
    cuda = params["parameters"][0]["cuda"]
    test_dataset = str(params["parameters"][0]["test_dataset"])
    model_ = str(params["parameters"][0]["model_"])
    # model_ = "dummy_model"

    #load training dataset
    cullpdb = CullPDB()

    all_models.append(model_)

    if model_ not in all_models:
        raise ValueError('Model must be in available models.')

    # module_import = ('import models.{} as model_mod'.format(model_))
    # print(module_import)
    # mod = importlib.import_module("models.")
    # model = model_mod.build_model(params)
    mod = importlib.import_module("models."+model_)
    print(mod)
    model = mod.build_model(params)

    # model = eval(model_).build_model(params)


    # create saved_models directory where trained models will be stored
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    #create logs path directory where TensorBoard logs will be stored
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    #fix checkpoints dir
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    #initialise keras callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'model_' + current_datetime + '.h5')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_accuracy', mode='max')
    stepDecay = StepDecay()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(stepDecay)

    start = time.time()

    # fit model
    if cuda:
        with tf.device('/gpu:0'):
            print('Fitting model...')
            history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm}, {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},{'main_output': cullpdb.vallabel}),
            epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, earlyStopping, checkpoint,lr_schedule],shuffle=True)
    else:
        print('Fitting model...')
        history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm}, {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},{'main_output': cullpdb.vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, earlyStopping, checkpoint,lr_schedule],shuffle=True)

    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}'.format(elapsed))

    model_output['Training Time'] = elapsed

    # save model locally in saved models dir - create dir in this dir to store all model related objects
    saved_model_path = os.path.join(os.getcwd(), OUTPUT_DIR)
    model_folder_path = os.path.join(saved_model_path, model_ + '_'+ current_datetime)
    os.makedirs(model_folder_path)
    save_path = os.path.join(model_folder_path, 'model.h5')
    print('Model saved in {} folder as {} '.format(model_folder_path, save_path))
    model.save(save_path)

    # save model history pickle
    history_filepath = os.path.join(model_folder_path, 'history.pckl')
    save_history(history, history_filepath)

    plot_history(history.history, model_folder_path, show_histograms = True,
        show_boxplots = True, show_kde = True, filter_outliers = True)

    #evaluating model
    evaluate_cullpdb(model,cullpdb)
    evaluate_model(model, test_dataset=test_dataset)

    #getting output results from model into csv
    get_model_output(model_folder_path)

    #save model architecture
    with open(os.path.join(model_folder_path, "model_architecture.json"), "w") as model_arch:
        model_arch.write(model.to_json(indent=3))

    append_model_output('Model Name', model_)
    append_model_output('Training Dataset  Type ', training_data)
    append_model_output('Number of epochs', epochs)
    append_model_output('Batch size', batch_size)
    append_model_output('TensorBoard logs dir', logs_path)
    append_model_output('Data directory', data_dir)

if __name__ == "__main__":

    #############################################################
    ###                   PSP Input Arguments                 ###
    #############################################################

    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')

    parser.add_argument('-config_', '--config_', type=str, required=True,
                        help='File path to config json file.')

    parser.add_argument('-job-dor', '--job-dir', help='GCS location to write checkpoints and export models',required=False)

    args = parser.parse_args()

    # main(args)
    main(args)
