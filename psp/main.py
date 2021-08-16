####################################
### Train CDULSTM/CDBLSTM models ###
####################################

#importing required modules and dependancies
import os, sys
from os import listdir
from os.path import isfile, join
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from datetime import date
from datetime import datetime
import time
import importlib
import json
from json.decoder import JSONDecodeError
from load_dataset import *
import models.psp_dcblstm_model as psp_dcblstm_model
import models.psp_dculstm_model as psp_dculstm_model
import models.dummy_model as dummy_model
import tests.test_model as test_model
import tests.test_datasets as test_datasets
from evaluate import *
from globals import *
import plot_model as plot_model
from utils import *

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
# tf.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
set_session(session)

#get model filenames from models directory
remove_py = lambda x: x[:-3]
all_models = list(map(remove_py,([f for f in listdir('models') if isfile(join('models', f)) and f[:3] == 'psp'] + \
                ([f for f in listdir(join('models','auxiliary_models')) if isfile(join('models','auxiliary_models', f)) \
                and f[:3] == 'psp']))))

#set tensorflow GPUOptions so TF doesn't overload GPU if present
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sesh = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))

# main function for building model
def main(args):

    """
    Description:
        Main function for training, evaluating and plotting PSP models
    Args:
        args (dict): parsed input arguments
    Returns:
        None
    """

    print(os.path.isfile(args.config))
    try:
        if not os.path.isfile(args.config):
            raise OSError('JSON config file not found at path: {}'.format(args.config))

        with open(args.config) as f:
            # f = open(args.config,)
            params = json.load(f)
    except JSONDecodeError as e:
        print('Error getting config JSON file: {}'.format(args.config))

    #get input arguments
    training_data = params["parameters"][0]["training_data"]
    filtered = params["parameters"][0]["filtered"]
    batch_size = int(params["parameters"][0]["batch_size"])
    epochs = int(params["parameters"][0]["epochs"])
    logs_path = str(params["parameters"][0]["logs_path"])
    data_dir = str(params["parameters"][0]["data_dir"])
    cuda = params["parameters"][0]["cuda"]
    test_dataset = str(params["parameters"][0]["test_dataset"])
    model_ = str(params["parameters"][0]["model_"])
    model_ = "dummy_model"

    #load training dataset
    cullpdb = CullPDB()

    all_models.append(model_)

    if model_ not in all_models:
        raise ValueError('Model must be in available models.')

    model = eval(model_).build_model(params)

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

    # plot history ##
    # if show_plots:
    plot_model.plot_history(history.history, model_folder_path, show_histograms = True,
        show_boxplots = True, show_kde = True, filter_outliers = True, save = True)

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

    parser.add_argument('-config', '--config', type=str, required=True,
                        help='')

    args = parser.parse_args()

    # main(args)
    main(args)
