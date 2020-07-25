#Main python file for training and testing PSP models


import os, sys
import numpy as np
import argparse
from data.get_dataset import download_export_dataset
from utils import *
from data.load_dataset import *
# PSP Arguments
# **************
parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
parser.add_argument('-b', '--batch_size', type=int, default=128,
                    help='batch size for training data (default: 42)')
# parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
#                     help='input batch size for testing (default: 1024)')
parser.add_argument('--data_dir', type=str, default='../data',
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



args = parser.parse_args()
