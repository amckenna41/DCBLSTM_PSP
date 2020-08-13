import os, sys
import numpy as np
import argparse
import subprocess
from models.psp_lstm_model import *
from data.load_dataset import *
from data.get_dataset import *
import tensorflow as tf
from tensorflow import keras

# from training.psp_lstm_gcp import *
# from psp_gcp.training.psp_lstm_gcp import *
# sys.path.append('..')
# from models.psp_lstm_model import *
# from models.psp_gru_model import *
# from data.load_dataset import *
# from data.get_dataset import *
# print(os.getcwd())
# os.chdir('..')
# models_path = os.getcwd() + '/' + 'models'
# data_path = os.getcwd() + '/' + 'data'
# sys.path.insert(1, models_path)
# sys.path.insert(1, data_path)
# print(os.getcwd())
# os.chdir('psp_gcp')

# # PSP Arguments
# # **************
# parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
# parser.add_argument('-b', '--batch_size', type=int, default=42,
#                     help='batch size for training data (default: 42)')
# # parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
# #                     help='input batch size for testing (default: 1024)')
# parser.add_argument('--data_dir', type=str, default='../data',
#                     help='Directory for training and test datasets')
# parser.add_argument('-sb','--storage_bucket', type=str, default='',
#                     help='Google Storage Bucket storing data and logs')
# # parser.add_argument('--result_dir', type=str, default='./result',
# #                     help='Output directory (default: ./result)')
# # parser.add_argument('--seed', type=int, default=1, metavar='S',
# #                     help='random seed (default: 1)')
# parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
#                     help ='The number of nodes for first LSTM hidden layer')
# parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
#                     help ='The number of nodes for second LSTM hidden layer')
# parser.add_argument('-dr', '--dropout', type=float, default = 0.5,
#                     help='The dropout applied to input (default = 0.5)')
# parser.add_argument('-op', '--optimizer', default = 'adam',
#                     help='The optimizer used in compiling and fitting the models')
# parser.add_argument('-e', '--epochs', type=int, default=10,
#                     help='The number of epochs to run on the model')
# parser.add_argument('jd', '--job_dir', help='GCS location to write checkpoints and export models',required=True)
# args = parser.parse_args()
#
# train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted()
#
# test_hot, testpssm, testlabel = load_cb513()
#
# d = os.getcwd()
# if d[len(d)-4:len(d)] == 'data':
#     os.chdir('..')
#     print(os.getcwd())
# def main(job_dir, **args):
def main(job_dir, args):
    logs_path = job_dir + 'logs/tensorboard'



    # sys.path.append('/Users/adammckenna/protein_structure_prediction_DeepLearning/data')
    # sys.path.append('/Users/adammckenna/protein_structure_prediction_DeepLearning/models')
    # sys.path.append("/root/.local/lib/python3.7/site-packages/models")
    # sys.path.append("/root/.local/lib/python3.7/site-packages/data")

    #do google cloud authentication for outside users calling function
    print(os.getcwd())
    # model = build_model_lstm()
    # model.save('model_1')
    # train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel = load_cul6133_filted()
    # print('Dataset loaded...')

    batch_size = str(args.batch_size)
    epochs = str(args.epochs)
    job_dir = str(args.job_dir)
    storage_bucket = str(args.storage_bucket)
    job_name = str(args.job_name)
    print(batch_size, epochs)
#    subprocess.call("chmod +x gcp_deploy.sh", shell=True)
    #create storage bucket if doesn't exist
    os.environ["BATCH_SIZE"] = batch_size
    os.environ["EPOCHS"]= epochs
    os.environ["JOB_DIR"] = job_dir
    os.environ["STORAGE_BUCKET"] = storage_bucket
    os.environ["REGION"] = "us-central1"
    os.environ["JOB_NAME"] = job_name
    # call(['bash', 'run.sh', batch_size, epochs])
    # process = subprocess.run('./gcp_deploy.sh',  check=True, timeout=10, env ={"BATCH_SIZE": batch_size, "EPOCHS":epochs})
    #chmod +x gcp_deploy.sh
    subprocess.call(["./psp_gcp/gcp_deploy.sh"],shell =True)
    # subprocess.Popen("./gcp_deploy.sh", shell =True, env ={"BATCH_SIZE": batch_size, "EPOCHS":epochs})
    print(os.getcwd())

    #load in data and create model, save model and import in psp_lstm_gcp
    #compile and fit model on cloud

    ##save keras model when built and before compiling it
##Running the app
if __name__ == "__main__":
    # PSP Arguments
    # **************
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('-b', '--batch_size', type=int, default=42,
                        help='batch size for training data (default: 42)')
    # parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
    #                     help='input batch size for testing (default: 1024)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory for training and test datasets')
    parser.add_argument('-sb','--storage_bucket', type=str, default='test_bucket',
                        help='Google Storage Bucket storing data and logs')
    # parser.add_argument('--result_dir', type=str, default='./result',
    #                     help='Output directory (default: ./result)')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('-lstm_1', '--lstm_layers1', type=int, default=400,
                        help ='The number of nodes for first LSTM hidden layer')
    parser.add_argument('-lstm_2', '--lstm_layers2', type=int, default=300,
                        help ='The number of nodes for second LSTM hidden layer')
    parser.add_argument('-dr', '--dropout', type=float, default = 0.5,
                        help='The dropout applied to input (default = 0.5)')
    parser.add_argument('-op', '--optimizer', default = 'adam',
                        help='The optimizer used in compiling and fitting the models')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to run on the model')
    parser.add_argument('-jd', '--job_dir', help='GCS location to write checkpoints and export models',required=False,
                        default = 'gs://keras-python-models')
    parser.add_argument('-job', '--job_name', help='Name of Keras training job',required=False,
                        default = 'JOB_1')
    args = parser.parse_args()

    # arguments = args.__dict__
    main(args.job_dir, args)

    # main(args.job_dir)
