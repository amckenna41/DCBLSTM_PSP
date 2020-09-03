#Main python file for training and evaluating PSP models locally

#importing required modules and dependancies
import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import subprocess
import tensorflow as tf
import subprocess
from data.load_dataset import *
from data.get_dataset import *
from models.psp_lstm_model import *
from models.psp_gru_model import *
from models.psp_lstm_pssm_model import *
from models.psp_lstm_seq_model import *
from models.plot_model import *
# from psp_gcp.training import *

#get model filenames from models directory
model_files = [f for f in listdir('models/') if isfile(join('models/', f))]
remove_py = lambda x: x[:-3]
model_files = list(map(remove_py, model_files))
model_files = [f for f in model_files if f[-5:] == 'model']

#set tensorflow GPUOptions so TF doesn't overload GPU if present
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sesh = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))

def main(args):

    all_data = int(args.alldata)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)
    data_dir = str(args.data_dir)
    show_plots = args.show_plots
    #load training dataset
    # train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)

    #build model
    model_ = str(args.model)

    if model_ == 'psp_gru_model':
        model = build_model_gru()
    elif model_ == 'psp_lstm_model':
        model = build_model_lstm()
    elif model_ == 'psp_lstm_pssm_model':
        model = build_lstm_pssm_model()
    elif model_ == 'psp_lstm_seq_model':
        model = build_lstm_seq_model()

    #create saved_models directory where trained models will be stored
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    #create logs path directory where TensorBoard logs will be stored
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    #initialise TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    #fit model
    # print('Fitting model...')
    # history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
    #     epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],shuffle=True)

    # #save model locally and to google cloud bucket
    print('Saving model')
    saved_model_path = os.getcwd() + '/' + 'saved_models/'
    model_folder_path = saved_model_path + model_ + '_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))
    os.makedirs(model_folder_path)

    save_path = saved_model_path + model_ + '_' + str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '.h5'

    print('Model saved in {} folder as {}: '.format(model_folder_path, save_path))
    model.save(save_path)

    #save model history pickle
    history_filepath = model_folder_path + '_' + str(datetime.date(datetime.now()))  +'.pckl'

    try:
        f = open(history_filepath, 'wb')
        pickle.dump(history.history, f)
        f.close()
    except pickle.UnpicklingError as e:
        print('Error', e)
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        print('Error creating history pickle')

    score = evaluate(model)
    print('Model Loss : ', score[0])
    print('Model Accuracy : ', score[1])

    # tensorboard --logdir models/logs - visualise model results on TensorBoard

    ## plot history ##
    if show_plots:
        models.plot_model(history.history, model_folder_path)


    # model_folder_path = 'saved_models/model_bgru_3x1Dconv'+ str(datetime.date(datetime.now()))
    # model_save_path = 'model_bgru_3x1Dconv_' +'epochs_' + str(args.epochs) +'_'+ 'batch_size_' + str(args.batch_size) + '_' + str(datetime.date(datetime.now())) + \
    #     '_' + str((datetime.now().strftime('%H:%M')))+ '_accuracy-'+ str(score[1]) \
    #     +'_loss-' + str(score[0]) + '.h5'
    #
    # model.save('../saved_models/' + model_save_path)
    # #create directory in bucket for new model - name it the model name, store model
    # plot_history(history.history, model_folder_path, show_histograms=True, show_boxplots=True, show_kde=True, save = True)



if __name__ == "__main__":

    # PSP Arguments
    # **************
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128,
                        help='batch size for training data (default: 128)')
    parser.add_argument('-batch_size_test', '--batch_size_test', type=int, default=1,
                        help='batch size for test data (default: 1)')
    parser.add_argument('-logs_dir', '--logs_dir', type=str, default='saved_models/logs',
                        help='Directory for Tensorboard logs to be stored, stored by default in saved_models/logs')
    parser.add_argument('-model_dir', '--model_dir', type=str, default='saved_models',
                        help='Directory for saving models logs, default in saved_models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory for training and test datasets, by default stored in data dir')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to run on the model')
    parser.add_argument('-model', '--model', choices=model_files, type=str.lower, default="psp_lstm_model",
                        help='Select what model from the models directory to build and train locally')
    parser.add_argument('-alldata', '--alldata', type =int, default=1,
                        help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc')
    parser.add_argument('-show_plots', '--show_plots', type =bool, default=True,
                        help='Select whether you want plots of the model history to show')

    args = parser.parse_args()

    print('Running main project script with following arguments')
    print('Model to be trained: ', args.model)
    print('Number of epochs: ', args.epochs)
    print('Proportion of training data to be used: ', args.alldata)
    print('Training Batch Size: ',int(args.batch_size))
    print('Test Batch Size: ', int(args.batch_size_test))
    print('TensorBoard Logs directory: ', args.logs_dir)
    print('Data stored in directory: ', args.data_dir)

    main(args)
