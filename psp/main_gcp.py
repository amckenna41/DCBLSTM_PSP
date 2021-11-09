################################################################################
#####  Entry script for psp_gcp dir for training on Google Cloud Platform  #####
################################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Input, Conv1D, \
    Embedding, Dense, Dropout, Activation,  Concatenate, Reshape,MaxPooling1D, Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, \
    ReduceLROnPlateau, LearningRateScheduler, CSVLogger
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, \
    MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
import os
from os.path import isfile, join
from os import listdir
import sys
from datetime import date
from datetime import datetime
import time
import importlib
import pkgutil
import json
from google.cloud import storage
from json.decoder import JSONDecodeError
from psp.load_dataset import *
from psp.plot_model import *
from psp.gcp_utils import *
from psp._globals import *
from psp.evaluate import *
from psp.models import *
from psp.models.auxiliary_models import *

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #reduce TF log output to only include Errors

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

#get model filenames from models and auxillary models directory
all_models =  [name for _, name, _ in pkgutil.iter_modules([os.path.join('psp','models')])]
all_models = all_models + [name for _, name, _ in pkgutil.iter_modules([os.path.join('psp','models','auxiliary_models')])]

#main function to train and evaluate CNN + RNN + DNN model
def main(args):
    """
    Description:
        Main function for training, evaluating and plotting PSP models via GCP.
    Args:
        :args (dict): parsed input arguments.
    Returns:
        None
    """
    #load json from config input parameters
    params = json.loads(args.params)
    gcp_params = json.loads(args.gcp_params)
    model_params = json.loads(args.model_params)

    #get input arguments
    config = args.config
    local = args.local
    job_dir = args.job_dir
    package_path = gcp_params["package_path"]
    bucket = gcp_params["bucket"]
    training_data = params["training_data"]
    filtered = params["filtered"]
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    logs_path = str(params["logs_path"])
    cuda = params["cuda"]
    test_dataset = str(params["test_dataset"])
    model_ = str(params["model"])
    tf_version = tf.__version__
    lr_scheduler = str(model_params["lr_scheduler"])
    callbacks = (model_params["callbacks"])

    #initialise global GCP bucket variable
    initialise_bucket(bucket)

    #create data dir to store all training and test datasets
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    #create output dir to store model training output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    #create folder where all model assets and artifacts will be stored after training
    model_output_folder = os.path.join(os.path.join(OUTPUT_DIR, model_ + '_'+ current_datetime))
    os.makedirs(model_output_folder)

    #create logs path directory where TensorBoard logs will be stored
    if not os.path.exists(os.path.join(model_output_folder, logs_path)):
        os.makedirs(os.path.join(model_output_folder, logs_path))

    #create checkpoints dir where model checkpoints will be saved
    if not os.path.exists(os.path.join(model_output_folder, 'checkpoints')):
        os.makedirs(os.path.join(model_output_folder, 'checkpoints'))

    #append parameters to model output results file
    model_output["Output Folder"] = model_output_folder
    model_output["Config"] = os.path.basename(config)
    model_output["Model"] = model_
    model_output["Bucket"] = bucket
    model_output["Training Dataset Type"] = training_data
    model_output["Filtered?"] = filtered
    model_output["Test Dataset"] = test_dataset
    model_output["Number of epochs"] = epochs
    model_output["Batch size"] = batch_size
    model_output["Tensorflow Version"] = tf_version
    model_output["TensorBoard logs dir"] = os.path.join(model_output_folder, logs_path)
    model_output["Cuda"] = cuda
    model_output["LR Scheduler"] = lr_scheduler

    #load training dataset
    cullpdb = CullPDB(type=training_data, filtered=filtered)

    all_models.append(model_)
    #verify model specified in config parameter is an available trainable model
    if model_ not in all_models:
        raise ValueError('Model must be in available models.')

    #import model module from models or auxillary models folder
    if (model_!="psp_dcblstm_model" and model_!="psp_dculstm_model" and model_!="dummy_model"):
        mod = importlib.import_module(package_path + ".models.auxiliary_models."+model_)
    else:
        mod = importlib.import_module(package_path + ".models."+model_)

    #build imported model with parameters from config
    model = mod.build_model(model_params)

    all_callbacks = []

    #initialise Tensorflow callbacks, append each callback if used
    if (callbacks["tensorboard"]):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=(os.path.join(model_output_folder,
            logs_path)), histogram_freq=0, write_graph=True, write_images=True)
        all_callbacks.append(tensorboard)
    if (callbacks["earlyStopping"]):
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        all_callbacks.append(earlyStopping)
    if (callbacks["modelCheckpoint"]):
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_output_folder, 'checkpoints','model_' + current_datetime + '.h5'), \
            verbose=1,save_best_only=True, monitor='val_accuracy', mode='max')
        all_callbacks.append(checkpoint)
    if (callbacks["csv_logger"]):
        csv_logger = CSVLogger(os.path.join(model_output_folder, 'training.log'))
        all_callbacks.append(csv_logger)

    #get LR Scheduler callback to use from parameter in config file
    #remove any whitespace or '-' from lr_schedule name
    lr_scheduler = lr_scheduler.lower().strip().replace(" ", "").replace("-","")
    if (lr_scheduler == "exceptionaldecay" or lr_scheduler == "exponential"):
        exponentialDecay = ExponentialDecay()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(exponentialDecay)
        all_callbacks.append(lr_schedule)
    elif (lr_scheduler == "timebaseddecay" or lr_scheduler == "timebased"):
        timeBasedDecay = TimedBased()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(timeBasedDecay)
        all_callbacks.append(lr_schedule)
    elif (lr_scheduler == "stepdecay" or lr_scheduler == "exponential"):
        stepDecay = StepDecay()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(stepDecay)
        all_callbacks.append(lr_schedule)

    #start counter
    start = time.time()

    #fit model
    if cuda:
        with tf.device('/gpu:0'):   #if training on GPU
            print('Fitting model...')
            history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm},
                {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},
                {'main_output': cullpdb.vallabel}), epochs=epochs, batch_size=batch_size, verbose=2,
                callbacks=all_callbacks,shuffle=True)
    else:   #training on CPU (default)
        print('Fitting model...')
        history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm},
            {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},
            {'main_output': cullpdb.vallabel}), epochs=epochs, batch_size=batch_size, verbose=2,
            callbacks=all_callbacks,shuffle=True)

    #stop counter, calculate elapsed time
    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}'.format(elapsed))
    model_output["Training Time"] = elapsed

    #save model locally in saved models dir - create dir in this dir to store all model related objects
    print('Model saved in {} folder as {} '.format(
        os.path.dirname(model_output_folder), os.path.basename(os.path.join(model_output_folder, 'model.h5'))))
    model.save(os.path.join(model_output_folder, 'model.h5'))

    #save model history pickle
    history_filepath = os.path.join(model_output_folder, 'history.pckl')
    save_history(history, history_filepath)

    #plot model history and all metric plots
    plot_history(history.history, model_output_folder, show_histograms = True,
        show_boxplots = True, show_kde = True, filter_outliers = True)

    #evaluating model on test datasets
    evaluate_cullpdb(model,cullpdb)
    evaluate_model(model, test_dataset=test_dataset)

    #visualise Keras model and all its layers, store in png
    #Need to manually install graphviz (https://graphviz.gitlab.io/download/) etc...
    if (local=="1"):
        visualise_model(model, model_output_folder)

    #save model architecture
    with open(os.path.join(model_output_folder, "model_architecture.json"), "w") as model_arch:
        model_arch.write(model.to_json(indent=3))

    #getting output results from model into csv
    model_output_df = get_model_output(model_output_folder)

    #upload configuration json to storage bucket
    #local flag used as config file upload doesn't seem to work when training on GCP, only locally
    if (local=="1"):
        upload_file(os.path.join(model_output_folder,os.path.basename(config)),config)

    # upload model output folder and all training results and assets
    upload_directory(model_output_folder, model_output_folder)

    print('Model training files exported to bucket path: {}/{} '.format(bucket, model_output_folder))

    #append training results of current job to all results file
    append_all_output(model_output_df)

    #close tensorflow session
    session.close()

if __name__ == "__main__":

    #############################################################
    ###                   PSP Input Arguments                 ###
    #############################################################

    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')

    parser.add_argument('-local', '--local', required=True,
                        help='Flag to determine if job being run locally or on GCP.')
    parser.add_argument('-job-dir', '--job-dir', type=str, required=True,
                        help='Directory where logs from training job are stored.')
    parser.add_argument('-config', '--config', type=str, required=True,
                        help='File path to config json file.')
    parser.add_argument('-params', '--params', type=str, required=True,
                        help='General training parameters')
    parser.add_argument('-gcp_params', '--gcp_params', type=str, required=True,
                        help='GCP job parameters')
    parser.add_argument('-model_params', '--model_params', type=str, required=True,
                        help='ML model parameters')

    #parse input arguments
    args = parser.parse_args()

    main(args)
