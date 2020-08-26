import numpy as np
import gzip
import h5py
import tensorflow as tf
import argparse
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io
import os
import sys
import importlib
from datetime import date
from datetime import datetime
from google.cloud import storage
import subprocess
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import training.gcp_utils as utils

BUCKET_NAME = 'keras-python-models'
# storage_client = storage.Client.from_service_account_json("psp-keras-training.json")

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)

# history_accuracy_array = np.zeros(0)
# history_val_accuracy_array = np.zeros(0)
#
# history_loss_array = np.zeros(0)
# history_val_loss_array = np.zeros(0)
#
# history_mse_array = np.zeros(0)
# history_val_mse_array = np.zeros(0)
#
# history_mae_array = np.zeros(0)
# history_val_mae_array = np.zeros(0)


def plot_history(history_filepath, show_histograms = False, show_boxplots = False,
                    show_kde = False):


    # f = open(history_filepath, 'rb')
    # history = pickle.load(f)
    # f.close()
    history = history_filepath

    history_df = pd.DataFrame(history.items(), columns =['Metrics','Score'], index = history.keys())
    del history_df['Metrics']
    history_df_trans = history_df.T

    global history_accuracy_array
    history_accuracy_array = np.array(history_df_trans['accuracy'][0])
    global history_val_accuracy_array
    history_val_accuracy_array = np.array(history_df_trans['val_accuracy'][0])

    global history_loss_array
    history_loss_array = np.array(history_df_trans['loss'][0])
    global history_val_loss_array
    history_val_loss_array = np.array(history_df_trans['val_loss'][0])

    global history_mse_array
    history_mse_array = np.array(history_df_trans['mean_squared_error'][0])
    global history_val_mse_array
    history_val_mse_array = np.array(history_df_trans['val_mean_squared_error'][0])

    global history_mae_array
    history_mae_array = np.array(history_df_trans['mean_absolute_error'][0])
    global history_val_mae_array
    history_val_mae_array = np.array(history_df_trans['val_mean_absolute_error'][0])

    global history_recall_array
    history_recall_array = np.array(history_df_trans['recall'][0])
    global history_val_recall_array
    history_val_recall_array = np.array(history_df_trans['val_recall'][0])

    global history_precision_array
    history_precision_array = np.array(history_df_trans['precision'][0])
    global history_val_precision_array
    history_val_precision_array = np.array(history_df_trans['val_precision'][0])

    # if not (os.path.isfile(history_filepath)):

    #check cwd for history.pickle

    # f = BytesIO(file_io.read_file_to_string('gs://keras-python-models/cullpdb+profile_6133_filtered.npy', binary_mode=True))

    # if history is:
    #     #check cwd for history.pickle
    #     f = open('history_2.pckl', 'rb')
    #     history_4 = pickle.load(f)
    #     f.close()

    accuracy_fig_filename = 'accuracy_fig'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    loss_fig_filename = 'loss_fig'+ str(datetime.date(datetime.now()))+ \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mae_fig_filename = 'mae_fig'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mse_fig_filename = 'mse_fig'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '.png'
    recall_fig_filename = 'recall_fig'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    precision_fig_filename = 'precision_fig'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M')))+ '.png'

    #plot train and validation accuracy on history
    plt.figure()
    plt.plot(history_accuracy_array)
    plt.plot(history_val_accuracy_array)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')
    plt.savefig(accuracy_fig_filename, dpi=200)
    plt.show()
    plt.close()

    #plot train and validation loss on history
    plt.figure()
    plt.plot(history_loss_array)
    plt.plot(history_val_loss_array)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig(loss_fig_filename, dpi=200)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history_mae_array)
    plt.plot(history_val_mae_array)
    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['train_mae', 'val_mae'], loc='upper left')
    plt.savefig(mae_fig_filename, dpi=200)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history_mse_array)
    plt.plot(history_val_mse_array)
    plt.title('Model Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['train_mse', 'val_mse'], loc='upper left')
    plt.savefig(mse_fig_filename, dpi=200)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history_recall_array)
    plt.plot(history_val_recall_array)
    plt.title('Training & Validation Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train_recall', 'val_recall'], loc='upper left')
    plt.savefig(recall_fig_filename, dpi=200)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history_precision_array)
    plt.plot(history_val_precision_array)
    plt.title('Training & Validation Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train_precision', 'val_precision'], loc='upper left')
    plt.savefig(precision_fig_filename, dpi=200)
    plt.show()
    plt.close()

    # blob_path = 'plots/accuracy_fig'+ str(datetime.date(datetime.now())) + '.png'
    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/accuracy_fig_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, accuracy_fig_filename)
    # blob = bucket.blob(blob_path)
    # blob.upload_from_filename(accuracy_fig_filename)

    # blob_path = 'plots/loss_fig'+ str(datetime.date(datetime.now())) + '.png'
    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/loss_fig_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, loss_fig_filename)

    # blob = bucket.blob(blob_path)
    # blob.upload_from_filename(loss_fig_filename)

    # blob_path = 'plots/mae_fig'+ str(datetime.date(datetime.now())) + '.png'
    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mae_fig_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mae_fig_filename)

    # blob = bucket.blob(blob_path)
    # blob.upload_from_filename(mae_fig_filename)


    # blob_path = 'plots/mse_fig'+ str(datetime.date(datetime.now())) + '.png'
    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mse_fig_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mse_fig_filename)

    #Recall & PRECISION upload s


    if (show_histograms):
        plot_histograms(history)

    if (show_boxplots):
        plot_boxplots(history)

    if (show_kde):
        plot_kde(history)

    ##RECALL & PRECISION GRAPHS##
    ##RECALL & PRECISION GRAPHS##
    ##RECALL & PRECISION GRAPHS##

def plot_boxplots(history):

    accuracy_box_filename = 'accuracy_boxplot_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    loss_box_filename = 'loss_boxplot_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mae_box_filename = 'mae_boxplot_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mse_box_filename = 'mse_boxplot_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'

    #Boxplots

    filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]

    # plt.boxplot(df_array[1][0], patch_artist=True)
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Accuracy"], fontsize = 15)
    plt.title('Boxplot of training accuracy', fontsize = 20)
    plt.savefig(accuracy_box_filename, dpi=200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/accuracy_boxplot_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, accuracy_box_filename)

    filtered = history_loss_array[~is_outlier(history_loss_array)]

    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Loss"], fontsize = 15)
    plt.title('Boxplot of training loss', fontsize = 20)
    plt.savefig(loss_box_filename, dpi=200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/loss_boxplot_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, loss_box_filename)

    filtered = history_mse_array[~is_outlier(history_mse_array)]

    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Squared Error"], fontsize = 15)
    plt.title('Boxplot of training mean squared error', fontsize = 20)
    plt.savefig(mse_box_filename, dpi=200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mse_boxplot_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mse_box_filename)

    filtered = history_mae_array[~is_outlier(history_mae_array)]

    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Absolute Error"], fontsize = 15)
    plt.title('Boxplot of training mean absolute error', fontsize = 20)
    plt.savefig(mae_box_filename, dpi=200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mae_boxplot_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mae_box_filename)


    #sns.pairplot

def plot_histograms(history):

    accuracy_hist_filename = 'accuracy_hist'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    loss_hist_filename = 'loss_hist'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mae_hist_filename = 'mae_hist'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mse_hist_filename = 'mse_hist'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'

    filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]
    val_filtered = history_val_accuracy_array[~is_outlier(history_val_accuracy_array)]

    #Add stddev
    #Training and validation accuracy histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_accuracy_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_accuracy_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Accuracy', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    # plt.xlim(0.7)
    accuracy_mean = "Train Accuracy Mean = {:.3f} \n Val Accuracy Mean = {:.3f}".format(history_accuracy_array.mean(), history_val_accuracy_array.mean())
    plt.text(0.7, 0.9, accuracy_mean, transform=plt.gca().transAxes, fontweight='bold')
    # plt.text(history_accuracy_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_accuracy_array.mean()), fontsize=15)
    # plt.text(history_val_accuracy_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_val_accuracy_array.mean()), fontsize=15)
    plt.title('Histogram of Accuracy & Validation Accuracy',fontsize=20)
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.axvline(history_accuracy_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_accuracy_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.savefig(accuracy_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/accuracy_hist_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, accuracy_hist_filename)

    filtered = history_loss_array[~is_outlier(history_loss_array)]
    val_filtered = history_val_loss_array[~is_outlier(history_val_loss_array)]

    #Training and validation loss histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_loss_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_loss_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Loss', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    # plt.xlim(0.7)
    loss_mean = "Train Loss Mean = {:.3f} \n Val Loss Mean = {:.3f}".format(history_loss_array.mean(), history_val_loss_array.mean())
    plt.text(0.75, 0.9, loss_mean, transform=plt.gca().transAxes, fontweight='bold')
    # plt.text(history_loss_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_loss_array.mean()), fontsize=15)
    # plt.text(history_val_loss_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_val_loss_array.mean()), fontsize=15)
    plt.title('Histogram of Loss & Validation Loss',fontsize=20)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.axvline(history_loss_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_loss_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.savefig(loss_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/loss_hist_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, loss_hist_filename)

    filtered = history_mae_array[~is_outlier(history_mae_array)]
    val_filtered = history_val_mae_array[~is_outlier(history_val_mae_array)]

    #Training and validation Mean Absolute Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_mae_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_mae_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Absolute Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    # plt.xlim(0.7)
    mae_mean = "Train MAE Mean = {:.3f} \n Val MAE Mean = {:.3f}".format(history_mae_array.mean(), history_val_mae_array.mean())
    plt.text(0.75, 0.9, mae_mean, transform=plt.gca().transAxes, fontweight='bold')
    # plt.text(history_mae_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_mae_array.mean()), fontsize=15)
    # plt.text(history_val_mae_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_val_mae_array.mean()), fontsize=15)
    plt.title('Histogram of Training & Validation MAE',fontsize=20)
    plt.legend(['mae', 'val_mae'], loc='upper left')
    plt.axvline(history_mae_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mae_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.savefig(mae_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mae_hist_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mae_hist_filename)

    filtered = history_mse_array[~is_outlier(history_mse_array)]
    val_filtered = history_val_mse_array[~is_outlier(history_val_mse_array)]

    #Training and validation Mean Squared Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_mse_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_mse_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Squared Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    # plt.xlim(0.7)
    mse_mean = "Train MSE Mean = {:.3f} \n Val MSE Mean = {:.3f}".format(history_mse_array.mean(), history_val_mse_array.mean())
    plt.text(0.75, 0.9, mse_mean, transform=plt.gca().transAxes, fontweight='bold')
    # plt.text(history_mse_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_mse_array.mean()), fontsize=15)
    # plt.text(history_val_mse_array.mean()*0.995,2.5,'Mean: {:.4f}'.format(history_val_mse_array.mean()), fontsize=15)
    plt.title('Histogram of Training & Validation MSE',fontsize=20)
    plt.legend(['mse', 'val_mse'], loc='upper left')
    plt.axvline(history_mse_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mse_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.savefig(mse_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mse_hist_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mse_hist_filename)

    # blob = bucket.blob(blob_path)
    # blob.upload_from_filename(loss_hist_filename)

    #
    # plt.figure(figsize=[20,20])
    # f,a = plt.subplots(2,2)
    # a = a.ravel()
    # for idx,ax in enumerate(a):
    #
    #     ax.hist(history_df_trans.iloc[:,idx][0], color='#0504aa',alpha=0.5, rwidth=0.85, bins = 5, orientation='vertical')
    #     ax.set_title(history_df_trans.columns[idx])
    #     ax.set_xlabel('Accuracy')
    #     ax.set_ylabel('Frequency')
    #     temp_hist_array = np.array(history_df_trans.iloc[:,idx][0])
    #     ax.axvline(temp_hist_array.mean(), color='red', linestyle='dashed',linewidth=2)
    # plt.tight_layout()
    # plt.savefig('accuracy_hist.png', dpi = 200)
    # plt.close()


def plot_kde(history):

    accuracy_kde_filename = 'accuracy_kde_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    loss_kde_filename = 'loss_kde_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mae_kde_filename = 'mae_kde_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'
    mse_kde_filename = 'mse_kde_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) + '.png'

    #Accuracy KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_accuracy_array, shade=True, color="b", label="accuracy", alpha=.5)
    sns.kdeplot(history_val_accuracy_array, shade=True, color="g", label="val_accuracy", alpha=.5)
    plt.title('KDE Plot for Training/Validation Accuracy', fontsize = 20)
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.savefig(accuracy_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/accuracy_kde_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, accuracy_kde_filename)

    #Loss KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_loss_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_loss_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Loss', fontsize = 20)
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.savefig(loss_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/loss_kde_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, loss_kde_filename)

    #Mean Absolute Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mae_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mae_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Absolute Error', fontsize = 17)
    plt.xlabel("Mean Absolute Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.savefig(mae_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mae_kde_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mae_kde_filename)

    #Mean Squared Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mse_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mse_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Squared Error', fontsize = 17)
    plt.xlabel("Mean Squared Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.savefig(mse_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = 'plots/plots_' + str(datetime.date(datetime.now())) + '/mse_kde_'+ \
        str(datetime.date(datetime.now())) + '_'+ str((datetime.now().strftime('%H:%M'))) + '.png'
    utils.upload_file(blob_path, mse_kde_filename)


def is_outlier(points, thresh=3):
    """
    https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
