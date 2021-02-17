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
from data.load_dataset import *
import models.psp_dcblstm_model as psp_dcblstm_model
import models.psp_dculstm_model as psp_dculstm_model
import tests.test_model as test_model
import tests.test_datasets as test_datasets
from evaluate import *
from globals import *
import plot_model as plot_model
from utils import *
import models.dummy_model as dummy_model

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

    #get input arguments
    training_data = args.training_data
    filtered = args.filtered
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)
    data_dir = str(args.data_dir)
    show_plots = args.show_plots
    cuda = args.cuda
    test_dataset = str(args.test_dataset)
    model_ = str(args.model)
    model_ = "dummy_model"
    #load training dataset
    cullpdb = CullPDB()

    all_models.append(model_)
    assert model_ in all_models, 'Model must be a model in models directory'

    # model = eval(model_).build_model()
    model_ = 'dummy_model'

    # module_import = ('import models.{} as model_mod'.format(model_))
    # module_import = 'models.dummy_model'
    module_import = 'models.dummy_model'

    # print(module_import)
    # exec module_import
    # model = model_mod.build_model()
    mod = importlib.import_module(module_import)

    epochs = 1
    model = mod.build_model()
    # test_datasets.run_dataset_tests()    #run tests

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
        print('Fitting model...')
        history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm}, {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},{'main_output': cullpdb.vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, earlyStopping, checkpoint,lr_schedule],shuffle=True)
        elapsed = (time.time() - start)
    else:
        print('Fitting model...')
        history = model.fit({'main_input': cullpdb.train_hot, 'aux_input': cullpdb.trainpssm}, {'main_output': cullpdb.trainlabel},validation_data=({'main_input': cullpdb.val_hot, 'aux_input': cullpdb.valpssm},{'main_output': cullpdb.vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, earlyStopping, checkpoint,lr_schedule],shuffle=True)

    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}'.format(elapsed))

    model_output['Training Time'] = elapsed

# with tf.device('/gpu:0'): #use for training with GPU on TF

    # save model locally in saved models dir - create dir in this dir to store all model related objects
    print('Saving model')
    saved_model_path = os.path.join(os.getcwd(), OUTPUT_DIR)

    model_folder_path = os.path.join(saved_model_path, model_ + '_'+ current_datetime)
    os.makedirs(model_folder_path)

    # save_path = os.path.join(model_folder_path, 'model.h5')
    # print('Model saved in {} folder as {} '.format(model_folder_path, save_path))
    # model.save(save_path)

    #save model history pickle
    # history_filepath = os.path.join(model_folder_path, 'history.pckl')
    # save_history(history, history_filepath)

    ## plot history ##
    # if show_plots:
    #     plot_model.plot_history(history.history, model_folder_path, show_histograms = True, show_boxplots = True, show_kde = True, filter_outliers = True, save = True)

    # #evaluating model
    # evaluate_cullpdb(model,cullpdb)
    # evaluate_model(model, test_dataset=test_dataset)

    #getting output results from model into csv
    get_model_output(model_folder_path)

    #save model architecture
    with open(os.path.join(model_folder_path, "model_architecture.json"), "w") as model_arch:
        model_arch.write(model.to_json(indent=3))

    # print(model.to_json(indent=4))

    #run model_function script


if __name__ == "__main__":

    #############################################################
    ###                   PSP Input Arguments                 ###
    #############################################################

    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')

    parser.add_argument('-batch_size', '--batch_size', type=int, default=120,
                        help='batch size for training data (default: 120)')

    parser.add_argument('-logs_dir', '--logs_dir', type=str, default=OUTPUT_DIR +'/logs',
                        help='Directory for Tensorboard logs to be stored, stored by default in saved_models/logs')

    parser.add_argument('-model_dir', '--model_dir', type=str, default=OUTPUT_DIR,
                        help='Directory for saving models logs, default in saved_models')

    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Directory for training and test datasets, by default stored in data dir')

    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to run on the model')

    parser.add_argument('-model', '--model', choices=all_models, type=str.lower, default="psp_cdblstm_model",
                        help='Select what model from the models directory to build and train locally')

    parser.add_argument('-training_data', '--training_data', type =int, default=6133,
                        help='Select what type of training dataset to use, 6133 or 5926, default:6133 ')

    parser.add_argument('-filtered', '--filtered', type =bool, default=True,
                        help='Select what type of training dataset to use, filtered or unfiltered, default: True/Filtered')

    parser.add_argument('-dataset_type', '--dataset_type', type =int, default=5926,
                        help='Select what training dataset type to use - 5926 or 6133 (default: 5926)')

    parser.add_argument('-test_dataset', '--test_dataset',
                        help='Select what test dataset to use for evaluation, default is CB513',required=False, default = "all")

    parser.add_argument('-show_plots', '--show_plots', type =bool, required=False, default=True,
                        help='Select whether you want plots of the model history to show')

    parser.add_argument('-cuda', '--cuda',
                       help='Enable CUDA to train using GPU; default is CPU',required=False, default = False)

    args = parser.parse_args()

    print('Running main project script with following arguments')
    print('Model to be trained: ', args.model)
    print('Number of epochs: ', args.epochs)
    print('Training Batch Size: ',int(args.batch_size))
    print('TensorBoard Logs directory: ', args.logs_dir)
    print('Data stored in directory: ', args.data_dir)

    append_model_output('Model Name', args.model)
    append_model_output('Training Dataset  Type ', args.dataset_type)
    append_model_output('Number of epochs', args.epochs)
    append_model_output('Batch size', args.batch_size)
    append_model_output('TensorBoard logs dir', args.logs_dir)
    append_model_output('Data directory', args.data_dir)

    # main(args)
    main(args)
