#Main python file for training and testing PSP models locally


import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import subprocess
import tensorflow as tf
from data.load_dataset import *
from data.get_dataset import *
from models.psp_lstm_model import *
from models.psp_gru_model import *
from models.psp_lstm_pssm_model import *
from models.psp_lstm_seq_model import *


model_files = [f for f in listdir('models/') if isfile(join('models/', f))]
remove_py = lambda x: x[:-3]
model_files = list(map(remove_py, model_files))
model_files = [f for f in model_files if f[-5:] == 'model']

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sesh = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
def main():

    all_data = int(args.alldata)

    #load training dataset
    train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted(all_data)
    #load testing dataset
    test_hot, testpssm, testlabel = load_cb513(all_data)

    # casp10_data_test_hot, casp10_data_pssm, test_labels = load_casp10()
    # casp11_data_test_hot, casp11_data_test_hot, test_labels = load_casp11()

    #build model
    model_ = str(args.model)
    if model_ == 'psp_gru_model':
        model = build_model_gru()
    elif model_ == 'psp_lstm_model':
        model = build_model_lstm()
    elif model == 'psp_lstm_pssm_model':
        model = build_lstm_pssm_model()
    elif model == 'psp_lstm_seq_model':
        model = build_lstm_seq_model()


    batch_size = int(args.batch_size)
    epochs = int(args.epochs)

    logs_path = str(args.logs_path)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    #fit model
    print('Fitting model...')
    history = model.fit({'main_input': train_hot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': val_hot, 'aux_input': valpssm},{'main_output': vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard,ReduceLROnPlateau()],shuffle=True)

    # #save model locally and to google cloud bucket
    print('Saving model')
    save_path = 'models/' + model_ + str(datetime.date(datetime.now())) + '.h5'
    model.save(save_path)

    # tensorboard --logdir models/logs - visualise model results on TensorBoard

    # plot_model(history)

def plot_model(history):

    #plot train and validation accuracy on history
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')
    plt.show()
    plt.close()

    #plot train and validation loss on history
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.show()
    plt.close()



# PSP Arguments
# **************
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='batch size for training data (default: 42)')
parser.add_argument('-logs_dir', '--logs_dir', type=str, default='models/logs',
                    help='Directory for Tensorboard logs to be stored')
# parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
#                     help='input batch size for testing (default: 1024)')
parser.add_argument('--data_dir', type=str, default='data',
                    help='Directory for training and test datasets')
# parser.add_argument('--result_dir', type=str, default='./result',
#                     help='Output directory (default: ./result)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('-dr', '--dropout', type=float, default = 0.5,
                    help='The dropout applied to input (default = 0.5)')
parser.add_argument('-op', '--optimizer', default = 'adam',
                    help='The optimizer used in compiling and fitting the models')
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help='The number of epochs to run on the model')
parser.add_argument('-model', '--model', choices=model_files, type=str.lower, default="psp_lstm_model",
                    help='Select what model from the models directory to build and train locally')
parser.add_argument('-alldata', '--alldata', type =int, default=1,
                    help='Select what proportion of training and test data to use, 1 - All data, 0.5 - 50%% of data etc')

args = parser.parse_args()

if __name__ == "__main__":

    print('Running ')

    main()

#
# train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted()
#
# test_hot, testpssm, testlabel = load_cb513()
#
# d = os.getcwd()
# if d[len(d)-4:len(d)] == 'data':
#     os.chdir('..')
#     print(os.getcwd())
#
# print(os.getcwd())
# os.chdir('psp_gcp')
# print(os.getcwd())
# process = subprocess.run('./gcp_deploy.sh', shell=True, check=True, timeout=10)
# os.chdir('..')

# process.wait() # Wait for process to complete.
#
# # iterate on the stdout line by line
# for line in process.stdout.readlines():
#     print(line)
#change directory to psp_gcp dir and call shell scipt
