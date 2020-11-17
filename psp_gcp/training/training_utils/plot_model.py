#Module for creating all relevant and useful plots from the metrics saved when training model

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
from training.training_utils.global_vars import *

#do docstrings for functions

# plt.grid()
#plot figures of metrics from history of model
def plot_history(history, model_folder_path,  show_histograms = False, show_boxplots = False, show_kde = False):

    #initialise all global variables used in plotting
    initialise_vars(history, model_folder_path)

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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close()

    #Upload all metric figures to GCP Storage
    blob_path = plots_path + str(datetime.date(datetime.now())) + '/accuracy_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, accuracy_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/loss_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, loss_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/accuracy_loss_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, accuracy_loss_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mae_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mae_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mse_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mse_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/recall_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, recall_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/precision_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, precision_fig_filename)

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/precision_recall_fig_'+ current_datetime + '.png'
    utils.upload_file(blob_path, precision_recall_fig_filename)

    #Plot histograms of metrics
    if (show_histograms):
        plot_histograms(history)

    #Plot boxplots of metrics
    if (show_boxplots):
        plot_boxplots(history)

    #Plot KDE of metrics
    if (show_kde):
        plot_kde(history)

#Plot Boxplots of history
def plot_boxplots(history):

    #filter outliers
    filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]

    #Boxplot of training accuracy
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Accuracy"], fontsize = 15)
    plt.title('Boxplot of training accuracy', fontsize = 20)
    plt.savefig(accuracy_box_filename, dpi=200)
    plt.show()
    plt.close()

    #Upload boxplot blob
    blob_path = plots_path + str(datetime.date(datetime.now())) + '/accuracy_boxplot_'+ current_datetime + '.png'
    utils.upload_file(blob_path, accuracy_box_filename)

    #filter outliers
    filtered = history_loss_array[~is_outlier(history_loss_array)]

    #Boxplot of training loss
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Loss"], fontsize = 15)
    plt.title('Boxplot of training loss', fontsize = 20)
    plt.savefig(loss_box_filename, dpi=200)
    plt.show()
    plt.close()

    #Upload boxplot blob
    blob_path = plots_path + str(datetime.date(datetime.now())) + '/loss_boxplot_'+ current_datetime + '.png'
    utils.upload_file(blob_path, loss_box_filename)

    #filter outliers
    filtered = history_mse_array[~is_outlier(history_mse_array)]

    #Boxplot of training MSE
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Squared Error"], fontsize = 15)
    plt.title('Boxplot of training mean squared error', fontsize = 20)
    plt.savefig(mse_box_filename, dpi=200)
    plt.show()
    plt.close()

    #Upload boxplot blob
    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mse_boxplot_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mse_box_filename)

    #filter outliers
    filtered = history_mae_array[~is_outlier(history_mae_array)]

    #Boxplot of training MAE
    plt.figure(figsize=[10,8])
    plt.boxplot(filtered, patch_artist=False)
    plt.xticks([1], ["Mean Absolute Error"], fontsize = 15)
    plt.title('Boxplot of training mean absolute error', fontsize = 20)
    plt.savefig(mae_box_filename, dpi=200)
    plt.show()
    plt.close()

    #Upload boxplot blob
    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mae_boxplot_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mae_box_filename)

    #sns.pairplot

#Plot histograms of metrics from model
def plot_histograms(history):

    #filter accuracy histograms for outliers
    filtered = history_accuracy_array[~is_outlier(history_accuracy_array)]
    val_filtered = history_val_accuracy_array[~is_outlier(history_val_accuracy_array)]

    #Training and validation accuracy histograms
    plt.figure(figsize=[10,8])
    plt.hist(filtered, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(val_filtered, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Accuracy', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    accuracy_mean = "Train Accuracy Mean = {:.3f} \n Val Accuracy Mean = {:.3f}".format(history_accuracy_array.mean(), history_val_accuracy_array.mean())
    plt.text(0.7, 0.9, accuracy_mean, transform=plt.gca().transAxes, fontweight='bold')
    plt.title('Histogram of Accuracy & Validation Accuracy',fontsize=20)
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.axvline(history_accuracy_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_accuracy_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(accuracy_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/accuracy_hist_'+ current_datetime + '.png'
    utils.upload_file(blob_path, accuracy_hist_filename)

    #filter loss histograms for outliers
    filtered = history_loss_array[~is_outlier(history_loss_array)]
    val_filtered = history_val_loss_array[~is_outlier(history_val_loss_array)]

    #Training and validation loss histograms
    plt.figure(figsize=[10,8])
    plt.hist(filtered, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(val_filtered, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Loss', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    loss_mean = "Train Loss Mean = {:.3f} \n Val Loss Mean = {:.3f}".format(history_loss_array.mean(), history_val_loss_array.mean())
    plt.text(0.75, 0.9, loss_mean, transform=plt.gca().transAxes, fontweight='bold')
    plt.title('Histogram of Loss & Validation Loss',fontsize=20)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.axvline(history_loss_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_loss_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(loss_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/loss_hist_'+ current_datetime + '.png'
    utils.upload_file(blob_path, loss_hist_filename)

    #filter MAE histograms for outliers
    filtered = history_mae_array[~is_outlier(history_mae_array)]
    val_filtered = history_val_mae_array[~is_outlier(history_val_mae_array)]

    #Training and validation Mean Absolute Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(filtered, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(val_filtered, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Absolute Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    mae_mean = "Train MAE Mean = {:.3f} \n Val MAE Mean = {:.3f}".format(history_mae_array.mean(), history_val_mae_array.mean())
    plt.text(0.75, 0.9, mae_mean, transform=plt.gca().transAxes, fontweight='bold')
    plt.title('Histogram of Training & Validation MAE',fontsize=20)
    plt.legend(['mae', 'val_mae'], loc='upper left')
    plt.axvline(history_mae_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mae_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(mae_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mae_hist_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mae_hist_filename)

    #filter MSE histograms for outliers
    filtered = history_mse_array[~is_outlier(history_mse_array)]
    val_filtered = history_val_mse_array[~is_outlier(history_val_mse_array)]

    #Training and validation Mean Squared Error histograms
    plt.figure(figsize=[10,8])
    plt.hist(filtered, facecolor='peru', edgecolor='blue',bins=10, alpha=0.5, orientation="vertical")
    plt.hist(val_filtered, facecolor='orangered', edgecolor='maroon',bins=10, alpha=0.5, orientation="vertical")
    plt.xlabel('Mean Squared Error', fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    mse_mean = "Train MSE Mean = {:.3f} \n Val MSE Mean = {:.3f}".format(history_mse_array.mean(), history_val_mse_array.mean())
    plt.text(0.75, 0.9, mse_mean, transform=plt.gca().transAxes, fontweight='bold')
    plt.title('Histogram of Training & Validation MSE',fontsize=20)
    plt.legend(['mse', 'val_mse'], loc='upper left')
    plt.axvline(history_mse_array.mean(), color='peru', linestyle='dashed',linewidth=2)
    plt.axvline(history_val_mse_array.mean(), color='orangered', linestyle='dashed',linewidth=2)
    plt.grid()
    plt.savefig(mse_hist_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mse_hist_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mse_hist_filename)

#Plot Kernel Density estimates of metrics from model
def plot_kde(history):

    #Accuracy KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_accuracy_array, shade=True, color="b", label="accuracy", alpha=.5)
    sns.kdeplot(history_val_accuracy_array, shade=True, color="g", label="val_accuracy", alpha=.5)
    plt.title('KDE Plot for Training/Validation Accuracy', fontsize = 20)
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(accuracy_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/accuracy_kde_'+ current_datetime + '.png'
    utils.upload_file(blob_path, accuracy_kde_filename)

    #Loss KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_loss_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_loss_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Loss', fontsize = 20)
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(loss_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/loss_kde_'+ current_datetime + '.png'
    utils.upload_file(blob_path, loss_kde_filename)

    #Mean Absolute Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mae_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mae_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Absolute Error', fontsize = 17)
    plt.xlabel("Mean Absolute Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(mae_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mae_kde_'+ current_datetime+ '.png'
    utils.upload_file(blob_path, mae_kde_filename)

    #Mean Squared Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mse_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mse_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Squared Error', fontsize = 17)
    plt.xlabel("Mean Squared Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    plt.savefig(mse_kde_filename, dpi = 200)
    plt.show()
    plt.close()

    blob_path = plots_path + str(datetime.date(datetime.now())) + '/mse_kde_'+ current_datetime + '.png'
    utils.upload_file(blob_path, mse_kde_filename)


#Function for calculating and removing outliers from an array of elements
def is_outlier(points, thresh=3):

    #Reference:
        #https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


#initialise global vars used for plotting
def initialise_vars(history,model_folder_path):

    #create dir for plots: model plots stored in same folder as trained model
    plots_path = model_folder_path + '/plots/plots_'

    # global current_datetime
    # current_datetime = str(datetime.date(datetime.now())) + \
    #     '_' + str((datetime.now().strftime('%H:%M')))

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
