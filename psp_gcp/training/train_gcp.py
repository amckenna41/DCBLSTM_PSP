#########################################################################
## Entry script for psp_gcp dir for training on Google Cloud Platform  ##
#########################################################################

#import required modules and dependancies
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Input, Conv1D, Embedding, Dense, Dropout, Activation,  Concatenate, Reshape,MaxPooling1D, Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import AUC, MeanSquaredError, FalseNegatives, FalsePositives, MeanAbsoluteError, TruePositives, TrueNegatives, Precision, Recall
from tensorflow.keras import activations
import os
from os.path import isfile, join
from os import listdir
import sys
from datetime import date
from datetime import datetime
import importlib
from training.training_utils.load_dataset import *
from training.training_utils.plot_model import *
from training.training_utils.gcp_utils import *
from training.training_utils.globals import *
from training.evaluate import *
from models import *
from models.auxiliary_models import *

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
all_models = ['psp_dcblstm_model', 'psp_dculstm_model', 'auxiliary_models.psp_cnn_model',\
'auxiliary_models.psp_dcbgru_model','auxiliary_models.psp_dcugru_model','auxiliary_models.psp_dnn_model', \
'auxiliary_models.psp_rbm_model','auxiliary_models.psp_rnn_model','dummy_model']

#main function to train and evaluate CNN + RNN + DNN model
def main(args):

    #setting parsed input arguments
    job_dir = str(args.job_dir)
    job_name = str(args.job_name)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    logs_path = str(args.logs_dir)
    test_dataset = str(args.test_dataset)
    model_ = str(args.model)
    show_plots = args.plots

    #load training dataset
    cullPDB = CullPDB()

    # build model
    print('Building {}', model_)
    model_ = 'dummy_model'
    # model_ = 'psp_dcblstm_model'

    assert model_ in all_models, ('{} model not in models directory'.format(model_))

    module_import = 'models.{}'.format(model_)
    mod = importlib.import_module(module_import)
    model = mod.build_model()

    #initialise model callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_path = "checkpoints_" + job_name + '.h5'
    checkpointer = ModelCheckpoint(filepath=checkpoint_path,verbose=1,save_best_only=True, monitor='val_accuracy', mode='max')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    lr_decay = LearningRateScheduler(StepDecay(initAlpha=0.0005, factor=0.8, dropEvery=10))

    epochs = 1
    batch_size = 256
    start = time.time()
# with tf.device('/gpu:0'): #use for training with GPU on TF
# with tf.device('/cpu:0'): #use for training with CPU on TF - Default

    print('Fitting model...')
    history = model.fit({'main_input': cullPDB.train_hot, 'aux_input': cullPDB.trainpssm}, {'main_output': cullPDB.trainlabel},validation_data=({'main_input': cullPDB.val_hot, 'aux_input': cullPDB.valpssm},{'main_output': cullPDB.vallabel}),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard, checkpointer,earlyStopping, lr_decay],shuffle=True)

    elapsed = (time.time() - start)
    print('Elapsed Training Time: {}\n'.format(elapsed))

    print("Training Accuracy: ", max(history.history['accuracy']))
    print("Training Loss: ", min(history.history['loss']))

    #adding model output hyperparameters to dataframe
    append_model_output('Job Name', job_name)
    append_model_output('Job Directory', job_dir)
    append_model_output('Batch Size', batch_size)
    append_model_output('Epochs', epochs)
    append_model_output('Epochs', epochs)
    append_model_output('Learning Rate',learning_r)
    append_model_output('TensorBoard Logs', logs_path)
    append_model_output('Elapsed Training Time',elapsed)
    append_model_output('Training Accuracy', max(history.history['accuracy']))
    append_model_output('Training Loss', max(history.history['loss']))
    append_model_output('Training MSE',max(history.history['mean_squared_error']))
    append_model_output('Training MAE',max(history.history['mean_absolute_error']))
    append_model_output('Training Recall', max(history.history['recall']))
    append_model_output('Training Precision', max(history.history['precision']))

    #save model locally and upload to GCP Storage in job_name directory
    print('Saving model')
    model.save('model.h5')
    upload_file(os.path.join(job_name, 'model.h5'), 'model.h5')

    #save model history locally and upload to GCP Storage in job_name directory
    save_history(history, job_name)

    #save model architecture in json format and upload to GCP Storage in job_name directory
    with open("model_architecture.json", "w") as model_arch:
        model_arch.write(model.to_json(indent=3))
    upload_file(os.path.join(job_name, "model_architecture.json"),"model_architecture.json")

    show_plots=False
    if show_plots:
        #visualise model and its metrics, storing and uploading plots to GCP Storage in job_name directory
        plot_history(history.history, job_name,show_histograms=True, show_boxplots=True, show_kde=True)

    #evaluating model
    evaluate_cullpdb(model,cullPDB)
    evaluate_model(model, test_dataset=test_dataset)

    #ouput results of model into csv, upload to GCP Storage in job_name directory
    get_model_output(job_name)


#############################################################
###                   PSP Input Arguments                 ###
#############################################################

parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')

parser.add_argument('-b', '--batch_size', type=int, default=120,
                    help='batch size for training data (default: 120)')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='The number of epochs to run on the model')

parser.add_argument('-jd', '--job-dir', help='GCS location to write checkpoints and export models',required=False,
                    default = "gs://" + BUCKET_NAME)

parser.add_argument('-j', '--job_name', help='Name of GCS Job',required=False,
                    default = "default_job_name")

parser.add_argument('-logs_dir', '--logs_dir',
                    help='Directory on cloud storage for Tensorboard logs',required=False, default = (BUCKET_NAME + "/logs-tensorboard"))

parser.add_argument('-test_dataset', '--test_dataset',
                    help='Select what test dataset to use for evaluation, default is CB513',required=False, default = "all")

parser.add_argument('-use_gpu', '--use_gpu',
                    help='Select whether to use a GPU for training',required=False, default = False)

parser.add_argument('-model', '--model',
                    help='Select what model to train on GCP',required=False, default = 'psp_dcblstm_model')

parser.add_argument('-plots', '--plots',
                    help='Select whether to create visualizations and plots',required=False, default = True)

args = parser.parse_args()


main(args)
