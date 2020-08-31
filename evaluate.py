
import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import tensorflow as tf

def evaluate(model):


    #load test dataset
    test_hot, testpssm, testlabel = load_cb513()

    #load CASP test datasets - Uncomment to test model on CASP10/11 Datasets
    # casp10_data_test_hot, casp10_data_pssm, test_labels = load_casp10()
    # casp11_data_test_hot, casp11_data_test_hot, test_labels = load_casp11()

    print('Evaluating model')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,batch_size=1)

    return score 
###Confusion matrix ###


#
#
# if __name__ == '__main__':
#
#     # PSP Arguments
#     # **************
#     parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
#     parser.add_argument('-batch_size_test', '--batch_size_test', type=int, default=1,
#                         help='batch size for test data (default: 1)')
#     parser.add_argument('-model', '--model', choices=model_files, type=str.lower, default="psp_lstm_model",
#                         help='Select what model from the models directory to build and train locally')
#     parser.add_argument('-test_dataset', '--test_dataset', type=str.lower, default="cb513",
#                         help='Select what test dataset to use for evaluating model')
#     args = parser.parse_args()
#
#     print('Evluating model ')
#
#     evaluate(args)
