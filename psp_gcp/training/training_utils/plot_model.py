
###################################################
### Plotting and visualsing metrics from models ###
###################################################

#import required modules and dependancies
import numpy as np
import pandas as pd
import os
import sys
from datetime import date
from datetime import datetime
from google.cloud import storage
import subprocess
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import training.training_utils.gcp_utils as utils
from training.training_utils.globals import *


def plot_history(history, job_name,  show_histograms = False, show_boxplots = False, show_kde = False, filter_outliers = True):

    """
    Description:
        Plotting and visualising basic figure plots from the model history
        trained on the training and validation datasets. Plotting following
        metrics: accuracy, loss, MSE, MAE, recall and precsion.
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        model_folder_path (str): path to model folder used to save plots and models to
        show_histograms (bool): visualise results via histograms, default: False.
        show_boxplots (bool): visualise results via boxplots, default: False.
        show_kde (bool): visualise results via kernel density plots, default: False.
        filter_outliers (bool): if True, use is_outlier() function to filter outliers from arrays, default: True
        save (bool): save visualisation in model folder, default: True
    Returns:
        None

    """

    #initialise all global variables used in plotting
    initialise_vars(history, job_name)

    #plot train and validation accuracy on history
    plt.figure()
    plt.plot(history_accuracy_array)
    plt.plot(history_val_accuracy_array)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
    plt.grid()
    plt.savefig(accuracy_fig_filename, dpi=200)
    plt.close()

    #plot train and validation loss & accuracy on history
    plt.figure()
    plt.plot(history_accuracy_array)
    plt.plot(history_val_accuracy_array)
    plt.plot(history_loss_array)
    plt.plot(history_val_loss_array)
    plt.title('Model Loss & Accuracy')
    plt.ylabel('Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss'], loc='upper left')
    plt.grid()
    plt.savefig(accuracy_loss_fig_filename, dpi=200)
    plt.close()

    #plot train and validation loss on history
    plt.figure()
    plt.plot(history_loss_array)
    plt.plot(history_val_loss_array)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.grid()
    plt.savefig(loss_fig_filename, dpi=200)
    plt.close()

    #plot train and validation Mean Absolute Error
    plt.figure()
    plt.plot(history_mae_array)
    plt.plot(history_val_mae_array)
    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['train_mae', 'val_mae'], loc='upper left')
    plt.grid()
    plt.savefig(mae_fig_filename, dpi=200)
    plt.close()

    #plot train and validation Mean Squared Error
    plt.figure()
    plt.plot(history_mse_array)
    plt.plot(history_val_mse_array)
    plt.title('Model Mean Squared Error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['train_mse', 'val_mse'], loc='upper left')
    plt.grid()
    plt.savefig(mse_fig_filename, dpi=200)
    plt.close()

    #plot train and validation Recall
    plt.figure()
    plt.plot(history_recall_array)
    plt.plot(history_val_recall_array)
    plt.title('Training & Validation Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train_recall', 'val_recall'], loc='upper left')
    plt.grid()
    plt.savefig(recall_fig_filename, dpi=200)
    plt.close()

    #plot train and validation Precision
    plt.figure()
    plt.plot(history_precision_array)
    plt.plot(history_val_precision_array)
    plt.title('Training & Validation Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train_precision', 'val_precision'], loc='upper left')
    plt.grid()
    plt.savefig(precision_fig_filename, dpi=200)
    plt.close()

    #plot train and validation Recall + Precision
    plt.figure()
    plt.plot(history_precision_array)
    plt.plot(history_val_precision_array)
    plt.plot(history_recall_array)
    plt.plot(history_val_recall_array)
    plt.title('Training and Validation Recall & Precision')
    plt.ylabel('Precision & Recall')
    plt.xlabel('Epoch')
    plt.legend(['train_precision', 'val_precision', 'train_recall', 'val_recall'], loc='upper left')
    plt.grid()
    plt.savefig(precision_recall_fig_filename, dpi=200)
    plt.close()

    #Upload all metric figures to GCP Storage
    blob_path = os.path.join(plots_path, 'accuracy_fig.png')
    utils.upload_file(blob_path, accuracy_fig_filename)

    blob_path = os.path.join(plots_path, 'loss_fig.png')
    utils.upload_file(blob_path, loss_fig_filename)

    blob_path = os.path.join(plots_path, 'accuracy_loss_fig.png')
    utils.upload_file(blob_path, accuracy_loss_fig_filename)

    blob_path = os.path.join(plots_path, 'mae_fig.png')
    utils.upload_file(blob_path, mae_fig_filename)

    blob_path = os.path.join(plots_path, 'mse_fig.png')
    utils.upload_file(blob_path, mse_fig_filename)

    blob_path = os.path.join(plots_path, 'recall_fig.png')
    utils.upload_file(blob_path, recall_fig_filename)

    blob_path = os.path.join(plots_path, 'precision_fig.png')
    utils.upload_file(blob_path, precision_fig_filename)

    blob_path = os.path.join(plots_path, 'precision_recall_fig.png')
    utils.upload_file(blob_path, precision_recall_fig_filename)

    #Plot histograms of metrics
    if (show_histograms):
        plot_histograms(history, filter_outliers)

    #Plot boxplots of metrics
    if (show_boxplots):
        plot_boxplots(history, filter_outliers)

    #Plot KDE of metrics
    if (show_kde):
        plot_kde(history, filter_outliers)


def plot_boxplots(history, filter_outliers):

    """
    Description:
        Plotting and visualising metrics using boxplots from the model history
        trained on the training and validation datasets. Boxplots can show the
        different quartiles, median, min & max values and any outliers. Plotting
        following metrics: accuracy, loss, MSE, MAE, recall and precsion.
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        save (bool): save visualisation in model folder, default: True
        filter_outliers (bool): filter out outliers of metric arrays before plotting
    Returns:
        None
    """

    #filter outliers
    if filter_outliers:
        filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]
    else:
        filtered = history_accuracy_array

    #Boxplot of training accuracy
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Accuracy"], fontsize = 15)
    plt.title('Boxplot of training accuracy', fontsize = 20)
    plt.grid()
    plt.savefig(accuracy_box_filename, dpi=200)
    plt.close()

    #Upload boxplot blob
    blob_path = os.path.join(plots_path, 'accuracy_boxplot.png')
    utils.upload_file(blob_path, accuracy_box_filename)

    #filter outliers
    if filter_outliers:
        filtered = history_loss_array[~is_outlier(history_loss_array)]
    else:
        filtered = history_loss_array

    #Boxplot of training loss
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Loss"], fontsize = 15)
    plt.title('Boxplot of training loss', fontsize = 20)
    plt.grid()
    plt.savefig(loss_box_filename, dpi=200)
    plt.close()

    #Upload boxplot blob
    blob_path = os.path.join(plots_path, 'loss_boxplot.png')
    utils.upload_file(blob_path, loss_box_filename)

    #filter outliers
    if filter_outliers:
        filtered = history_mse_array[~is_outlier(history_mse_array)]
    else:
        filtered = history_mse_array

    #Boxplot of training MSE
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Squared Error"], fontsize = 15)
    plt.title('Boxplot of training mean squared error', fontsize = 20)
    plt.grid()
    plt.savefig(mse_box_filename, dpi=200)
    plt.close()

    #Upload boxplot blob
    blob_path = os.path.join(plots_path, 'mse_boxplot.png')
    utils.upload_file(blob_path, mse_box_filename)

    #filter outliers
    if filter_outliers:
        filtered = history_mae_array[~is_outlier(history_mae_array)]
    else:
        filtered = history_mae_array

    #Boxplot of training MAE
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Absolute Error"], fontsize = 15)
    plt.title('Boxplot of training mean absolute error', fontsize = 20)
    plt.grid()
    plt.savefig(mae_box_filename, dpi=200)
    plt.close()

    #Upload boxplot blob
    blob_path = os.path.join(plots_path, 'mae_boxplot.png')
    utils.upload_file(blob_path, mae_box_filename)


def plot_histograms(history, filter_outliers):

    """
    Description:
        Plotting and visualising metrics using histograms from the model history
        trained on the training and validation datasets. Plotting following
        metrics: accuracy, loss, MSE, MAE, recall and precsion.
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        save (bool): save visualisation in model folder, default: True
        filter_outliers (bool): filter out outliers of metric arrays before plotting
    Returns:
        None
    """

    #filter accuracy histograms for outliers
    if filter_outliers:
        filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]
        val_filtered = history_val_accuracy_array[~is_outlier(history_val_accuracy_array)]
    else:
        filtered = history_accuracy_array
        val_filtered = history_val_accuracy_array

    #Training and validation accuracy histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_accuracy_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_accuracy_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Accuracy', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Histogram of Accuracy & Validation Accuracy',fontsize=20)
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.axvline(history_accuracy_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_accuracy_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(accuracy_hist_filename, dpi = 200)
    plt.close()

    blob_path = os.path.join(plots_path, 'accuracy_hist.png')
    utils.upload_file(blob_path, accuracy_hist_filename)

    #filter loss histograms for outliers
    if filter_outliers:
        filtered = history_loss_array[~is_outlier(history_loss_array)]
        val_filtered = history_val_loss_array[~is_outlier(history_val_loss_array)]
    else:
        filtered = history_loss_array
        val_filtered = history_val_loss_array

    #Training and validation loss histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_loss_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_loss_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Loss', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Histogram of Loss & Validation Loss',fontsize=20)
    plt.legend(['train_loss', 'train_loss'], loc='upper left')
    plt.axvline(history_loss_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_loss_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(loss_hist_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'loss_hist.png')
    utils.upload_file(blob_path, loss_hist_filename)

    #filter MAE histograms for outliers
    if filter_outliers:
        filtered = history_mae_array[~is_outlier(history_mae_array)]
        val_filtered = history_val_mae_array[~is_outlier(history_val_mae_array)]
    else:
        filtered = history_mae_array
        val_filtered = history_val_mae_array

    #Training and validation Mean Absolute Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_mae_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_mae_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Absolute Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Histogram of Training & Validation MAE',fontsize=20)
    plt.legend(['train_mae', 'val_mae'], loc='upper left')
    plt.axvline(history_mae_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mae_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(mae_hist_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'mae_hist.png')
    utils.upload_file(blob_path, mae_hist_filename)

    #filter MSE histograms for outliers
    if filter_outliers:
        filtered = history_mse_array[~is_outlier(history_mse_array)]
        val_filtered = history_val_mse_array[~is_outlier(history_val_mse_array)]
    else:
        filtered = history_mse_array
        val_filtered = history_val_mse_array

    #Training and validation Mean Squared Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(history_mse_array, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(history_val_mse_array, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Squared Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Histogram of Training & Validation MSE',fontsize=20)
    plt.legend(['train_mse', 'val_mse'], loc='upper left')
    plt.axvline(history_mse_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mse_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(mse_hist_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'mse_hist.png')
    utils.upload_file(blob_path, mse_hist_filename)


def plot_kde(history, filter_outliers):

    """
    Description:
        Plotting and visualising metrics using kernel density estimates from
        the model history trained on the training and validation datasets.
        Plotting following metrics: accuracy, loss, MSE, MAE, recall and precsion.
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        save (bool): save visualisation in model folder, default: True
        filter_outliers (bool): filter out outliers of metric arrays before plotting
    Returns:
        None

    """

    #Accuracy KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_accuracy_array, shade=True, color="b", label="accuracy", alpha=.5)
    sns.kdeplot(history_val_accuracy_array, shade=True, color="g", label="val_accuracy", alpha=.5)
    plt.title('KDE Plot for Training/Validation Accuracy', fontsize = 20)
    plt.legend(['train_acc', 'val_acc'], loc='upper right')
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(accuracy_kde_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'accuracy_kde.png')
    utils.upload_file(blob_path, accuracy_kde_filename)

    #Loss KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_loss_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_loss_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Loss', fontsize = 20)
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.savefig(loss_kde_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'loss_kde.png')
    utils.upload_file(blob_path, loss_kde_filename)

    #Mean Absolute Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mae_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mae_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Absolute Error', fontsize = 17)
    plt.legend(['train_mae', 'val_mae'], loc='upper right')
    plt.xlabel("Mean Absolute Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(mae_kde_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'mae_kde.png')
    utils.upload_file(blob_path, mae_kde_filename)

    #Mean Squared Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mse_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mse_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Squared Error', fontsize = 17)
    plt.legend(['train_mse', 'train_mse'], loc='upper right')
    plt.xlabel("Mean Squared Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(mse_kde_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'mse_kde.png')
    utils.upload_file(blob_path, mse_kde_filename)


def plot_protein_labels(pred_labels, calling_test_dataset):

    """
    Description:
        Plotting and visualising the distribution of the protein structure class labels
        after evaluation on the test datasets.
    Args:
        pred_labels (np.array): array of the predicted class labels from the test datasets
        calling_test_dataset (str): name of the test dataset calling function, used for labelling plots
    Returns:
        None
    """
    #plots_path ERROR

    protein_labels_filename = 'protein_labels_' + calling_test_dataset + '.png'
    print('Plots path',plots_path)
    print('Protein labels filename', protein_labels_filename)
    print('Lable new filepath : ', os.path.join(plots_path, protein_labels_filename))

    #converting the array of predicted labels and the values into Pandas Series
    protein_series = pd.Series([pred_labels[5], pred_labels[2],pred_labels[0], pred_labels[7],
        pred_labels[6], pred_labels[3], pred_labels[1], pred_labels[4]], ['H','E','L','T','S','G','B','I'])

    #colours for each label
    plot_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c','magenta']

    #plot horizontal barplot
    ax = protein_series.plot(kind='barh',color=plot_colors)
    plt.title('Protein Structure Labels after evaluation on '+calling_test_dataset, fontsize = 17)
    plt.xlabel('Label Proportion', fontsize = 15)
    plt.ylabel('Protein Label', fontsize = 15)
    plt.grid()
    plt.savefig(protein_labels_filename, dpi=200)
    plt.close()

    blob_path = os.path.join(plots_path, 'protein_labels.png')
    utils.upload_file(blob_path, protein_labels_filename)


def plot_lr(history):

    """
    Description:
        Plotting and visualising the learning rate during training of model.
    Args:
        history (dict): model training history
    Returns:
        None
    """
    learning_rate = history.history['lr']
    epochs = range(1, len(learning_rate) + 1)
    plt.plot(epochs, learning_rate)
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.grid()
    plt.savefig('lr_epochs.png', dpi=200)
    plt.close()

def is_outlier(points, thresh=3):

    """ Description:
            Auxillary function for calculting and removing outliers from array of elements
        Args:
            points(): An numobservations by numdimensions array of observations
            thread(int): The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers ,default: 3.
        Returns:
            mask : A numobservations-length boolean array.
         Reference:
            https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def initialise_vars(history,job_name):

    """
    Description:
        Initialise any global vars or global numpy arrays used for storing metrics from model history
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        model_folder_path (str): path to model folder used to save plots and models to
    Returns:
        None

    """
    #create dir for plots: model plots stored in same folder as trained model
    global plots_path
    plots_path = os.path.join(job_name, 'plots')

    #converting history into dataframe
    history_df = pd.DataFrame(history.items(), columns =['Metrics','Score'], index = history.keys())
    del history_df['Metrics']
    history_df_trans = history_df.T     #transpose dataframe

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
