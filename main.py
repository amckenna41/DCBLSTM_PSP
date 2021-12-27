################################################################################
#########                   Train PSP models                           #########
################################################################################

#importing required modules and dependancies
import os
import sys
from os import listdir
from os.path import join, isfile
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1.keras.backend import set_session
import time
import importlib
import json
from json.decoder import JSONDecodeError
from psp.load_dataset import *
from psp._globals import model_output, OUTPUT_DIR, current_datetime
from psp.plot_model import *
from psp.evaluate import *
from psp.utils import *
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #reduce TF log output to only include Errors

### Tensorboard parameters and configuration ###
tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()  # For easy reset of notebook state.
config_proto = tf.compat.v1.ConfigProto()
config_proto.allow_soft_placement = True
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.gpu_options.allow_growth = True
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
#set tensorflow GPUOptions so TF doesn't overload GPU if present
# config_proto.gpu_options(per_process_gpu_memory_fraction=0.333)
session = tf.compat.v1.Session(config=config_proto)
# tf.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
set_session(session)

#get model filenames from models directory
def remove_py(x): return os.path.splitext(x)[0]
# remove_py = lambda x: os.path.splitext(x)[0]
all_models = list(map(remove_py,([f for f in listdir(join('psp','models')) if isfile(join('psp','models', f)) and f[:3] == 'psp'] +
                ([f for f in listdir(join('psp','models','auxiliary_models')) if isfile(join('psp','models','auxiliary_models', f)) \
                and f[:3] == 'psp']))))
all_models.append('dummy_model')

#main starting function for PSP code pipeline
def main(args):
    """
    Description:
        Main function for training, evaluating and plotting PSP models.
    Args:
        :args (dict): parsed input arguments.
    Returns:
        None
    """
    #append config filepath to user input config file
    #strip file extension (if exists) from input filename and append .json to it
    config_file = os.path.join("config",os.path.splitext(args.config)[0]+'.json')

    #open JSON config file in config folder
    try:
        if not os.path.isfile(config_file):
            raise OSError('JSON config file not found at path: {}.'.format(config_file))
        with open(config_file) as f:
            params = json.load(f)
    except JSONDecodeError as e:
        print('Error getting config JSON file: {}.'.format(config_file))

    #parse input arguments from json config file
    training_data = params["parameters"][0]["training_data"]
    filtered = params["parameters"][0]["filtered"]
    batch_size = int(params["parameters"][0]["batch_size"])
    epochs = int(params["parameters"][0]["epochs"])
    learning_rate = float(params["model_parameters"][0]["optimizer"]["learning_rate"])
    logs_path = str(params["parameters"][0]["logs_path"])
    cuda = params["parameters"][0]["cuda"]
    test_dataset = str(params["parameters"][0]["test_dataset"])
    model_ = str(params["parameters"][0]["model"])
    tf_version = tf.__version__
    lr_scheduler = str(params["model_parameters"][0]["lr_scheduler"])
    callbacks = (params["model_parameters"][0]["callbacks"])

    #set model output dict to values in config
    model_output["Config"] = os.path.basename(config_file)
    model_output["Model"] = model_
    model_output["Training Dataset Type"] = training_data
    model_output["Filtered?"] = filtered
    model_output["Test Dataset"] = test_dataset
    model_output["Number of epochs"] = epochs
    model_output["Batch size"] = batch_size
    model_output["Tensorflow Version"] = tf_version
    model_output["Cuda"] = cuda
    model_output["LR Scheduler"] = lr_scheduler

    print("\n###################################################################")
    print("Running model locally with parameters...\n")
    print("Training Data: {} (filtered: {})".format(training_data, filtered))
    print("Test Dataset: {}".format(test_dataset))
    print("Model: {}".format(model_))
    print("Batch Size: {}".format(batch_size))
    print("Epochs: {}".format(epochs))
    print("Learning Rate: {}".format(learning_rate))
    print("Learning Rate Scheduler: {}".format(lr_scheduler))
    print("Logs Path: {}".format(logs_path))
    print("Cuda: {}".format(cuda))
    print("###################################################################\n")

    #verify model specified in config file exists in available models
    if model_ not in all_models:
        raise ValueError('Model {} must be in available models: \n {}.'.format(
            model_, all_models))

    #load cullPDB training dataset
    cullpdb = CullPDB(type=training_data, filtered=filtered)

    #import model module from models or auxillary models folder
    if (model_ != "psp_dcblstm_model" and model_ != "psp_dculstm_model" and model_ != "dummy_model"):
        mod = importlib.import_module("psp.models.auxiliary_models."+model_)
    else:
        mod = importlib.import_module("psp.models."+model_)

    #build model
    model = mod.build_model(params["model_parameters"][0])

    #create saved_models directory where trained models will be stored
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    #create folder where all model assets and artifacts will be stored after training
    model_folder_path = os.path.join(os.path.join(os.getcwd(), OUTPUT_DIR),
        model_ + '_'+ current_datetime)
    os.makedirs(model_folder_path)

    #create logs path directory where TensorBoard logs will be stored
    if not os.path.exists(os.path.join(model_folder_path, logs_path)):
        os.makedirs(os.path.join(model_folder_path, logs_path))

    #create checkpoints dir where model checkpoints will be saved
    if not os.path.exists(os.path.join(model_folder_path, 'checkpoints')):
        os.makedirs(os.path.join(model_folder_path, 'checkpoints'))

    all_callbacks = []

    #initialise Tensorflow callbacks
    #append each callback if used
    if (int(callbacks["tensorboard"])):
        tensorboard = TensorBoard(log_dir=(os.path.join(model_folder_path,
            logs_path)), histogram_freq=0, write_graph=True, write_images=True)
        all_callbacks.append(tensorboard)
    if (callbacks["earlyStopping"]):
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        all_callbacks.append(earlyStopping)
    if (callbacks["modelCheckpoint"]):
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_folder_path, 'checkpoints','model_' + current_datetime + '.h5'), \
            verbose=1,save_best_only=True, monitor='val_accuracy', mode='max')
        all_callbacks.append(checkpoint)
    if (callbacks["csv_logger"]):
        csv_logger = CSVLogger(os.path.join(model_folder_path, 'training.log'))
        all_callbacks.append(csv_logger)
    if (callbacks["reduceLROnPlateau"]):
        reduceLROnPlateau = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=10, verbose=1, mode="min")
        all_callbacks.append(reduceLROnPlateau)

    #get LR Scheduler callback to use from parameter in config file
    #remove any whitespace or '-' from lr_schedule name
    lr_scheduler = lr_scheduler.lower().strip().replace(" ", "").replace("-","")
    if (lr_scheduler == "exceptionaldecay" or lr_scheduler == "exponential"):
        exponentialDecay = ExponentialDecay()
        lr_schedule = LearningRateScheduler(exponentialDecay)
        all_callbacks.append(lr_schedule)
    elif (lr_scheduler == "timebaseddecay" or lr_scheduler == "timebased"):
        timeBasedDecay = TimedBased()
        lr_schedule = LearningRateScheduler(timeBasedDecay)
        all_callbacks.append(lr_schedule)
    elif (lr_scheduler == "stepdecay" or lr_scheduler == "exponential"):
        stepDecay = StepDecay()
        lr_schedule = LearningRateScheduler(stepDecay)
        all_callbacks.append(lr_schedule)

    #start time func to measure the training time
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

    #calculate elapsed training time
    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}'.format(elapsed))
    #append training time to output file
    model_output['Training Time'] = elapsed

    #save trained model
    model.save(os.path.join(model_folder_path, 'model.h5'))

    #save model history pickle
    save_history(history, os.path.join(model_folder_path, 'history.pckl'))

    #plot history
    plot_history(history.history, model_folder_path, show_histograms = True,
        show_boxplots = True, show_kde = True, filter_outliers = True, save = True)

    #visualise Keras model and all its layers, store in png in output folder
    visualise_model(model, model_folder_path)

    #evaluating model
    evaluate_cullpdb(model,cullpdb)
    evaluate_model(model, test_dataset=test_dataset)

    #getting output results from model into csv
    model_output_df = get_model_output(model_folder_path)

    #save model architecture
    with open(os.path.join(model_folder_path, "model_architecture.json"), "w") as model_arch:
        model_arch.write(model.to_json(indent=3))

    print('Model training files exported to local path: {} '.format(model_folder_path))

    #close tensorflow session
    session.close()

if __name__ == "__main__":

    #############################################################
    ###                   PSP Input Arguments                 ###
    #############################################################

    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction.')

    parser.add_argument('-config', '--config', type=str, required=True,
                        help='File path to config json file.')

    #parse input args
    args = parser.parse_args()

    main(args)
