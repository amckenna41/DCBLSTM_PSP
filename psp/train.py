####################################
### Train CDULSTM/CDBLSTM models ###
####################################

#importing required modules and dependancies
import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import subprocess
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
import subprocess
from datetime import date
from datetime import datetime
import time
from data.load_dataset import *
import models.psp_blstm_model as psp_blstm_model
import models.psp_ulstm_model as psp_ulstm_model
import tests.test_model as test_model
from globals import *

#get model filenames from models directory
model_files = [f for f in listdir('models') if isfile(join('models', f))]
remove_py = lambda x: x[:-3]
model_files = list(map(remove_py, model_files))
# model_files = [f for f in model_files if f[-5:] == 'model']
model_files = [f for f in model_files if f[:2] == 'psp']

#set tensorflow GPUOptions so TF doesn't overload GPU if present
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sesh = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))


# main function for building model
def main(args):

    #get input arguments
    all_data = int(args.alldata)
    filtered = args.filtered
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)
    data_dir = str(args.data_dir)
    show_plots = args.show_plots
    cuda = args.cuda
    test_dataset = str(args.test_dataset)

    #load training dataset

    #build model
    model_ = str(args.model)
    cul6133 = CulPDB6133(all_data, filtered)

    if model_ == 'psp_blstm_model':
        model = psp_blstm_model.build_model()
    elif model_ == 'psp_ulstm_model':
        model = psp_ulstm_model.build_model()

    # test_model.run_tests()    #run tests

    #create saved_models directory where trained models will be stored
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    #create logs path directory where TensorBoard logs will be stored
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    #initialise keras callbacks
    (os.path.join(os.getcwd(), 'data', train_path))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'model_' + current_datetime + '.h5')
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_acc', mode='max')

    start = time.time()

    #fit model
    if cuda:
        print('Fitting model...')
        history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, EarlyStopping, ModelCheckpoint, CSVLogger],shuffle=True)
        elapsed = (time.time() - start)
    else:
        print('Fitting model...')
        history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard, EarlyStopping, ModelCheckpoint, CSVLogger],shuffle=True)

    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}'.format(elapsed))

    start = time.time()
# with tf.device('/gpu:0'): #use for training with GPU on TF
    print('Fitting model...')

    elapsed = (time.time() - start)

    # #save model locally and to google cloud bucket
    print('Saving model')
    saved_model_path = os.getcwd() + '/' + 'saved_models/'
    model_folder_path = saved_model_path + model_ + '_'+ current_datetime
    os.makedirs(model_folder_path)

    save_path = saved_model_path + model_ + '_' + current_datetime + '.h5'

    print('Model saved in {} folder as {}: '.format(model_folder_path, save_path))
    # model.save(save_path)

    #save model history pickle
    history_filepath = model_folder_path + '_' + str(datetime.date(datetime.now()))  +'.pckl'


    ## plot history ##
    # if show_plots:
    #     models.plot_model(history.history, model_folder_path)

    # plot_main(history.history, model_folder_path, show_histograms=True, show_boxplots=True, show_kde=True, save = True)

    ## evaluate function/module

    get_model_output()







if __name__ == "__main__":

    # PSP Arguments
    ###############################

    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')

    parser.add_argument('-batch_size', '--batch_size', type=int, default=120,
                        help='batch size for training data (default: 120)')

    parser.add_argument('-logs_dir', '--logs_dir', type=str, default='saved_models/logs',
                        help='Directory for Tensorboard logs to be stored, stored by default in saved_models/logs')

    parser.add_argument('-model_dir', '--model_dir', type=str, default='saved_models',
                        help='Directory for saving models logs, default in saved_models')

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for training and test datasets, by default stored in data dir')

    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to run on the model')

    parser.add_argument('-model', '--model', choices=model_files, type=str.lower, default="psp_blstm_model",
                        help='Select what model from the models directory to build and train locally')

    parser.add_argument('-alldata', '--alldata', type =int, default=1,
                        help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc')

    parser.add_argument('-filtered', '--filtered', type =bool, default=True,
                        help='Select what type of training dataset to use, filtered or unfiltered, default: True/Filtered')

    parser.add_argument('-test_dataset', '--test_dataset',
                        help='Select what test dataset to use for evaluation, default is CB513',required=False, default = "CB513")

    parser.add_argument('-show_plots', '--show_plots', type =bool, default=True,
                        help='Select whether you want plots of the model history to show')

    parser.add_argument('-cuda', '--cuda',
                       help='Enable CUDA to train using GPU; default is CPU',required=False, default = False)

    args = parser.parse_args()

    print('Running main project script with following arguments')
    print('Model to be trained: ', args.model)
    print('Number of epochs: ', args.epochs)
    print('Proportion of training data to be used: ', args.alldata)
    print('Training Batch Size: ',int(args.batch_size))
    print('Test Batch Size: ', int(args.batch_size_test))
    print('TensorBoard Logs directory: ', args.logs_dir)
    print('Data stored in directory: ', args.data_dir)

    model_output['']
    model_output['']
    model_output['']
    model_output['']
    model_output['']
    model_output['']
    model_output['']

    main(args)
