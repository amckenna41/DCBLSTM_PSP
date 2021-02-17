###################################################
### Plotting and visualsing metrics from models ###
###################################################

#import required modules and dependancies
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from io import BytesIO
import os
import sys
from datetime import date
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from globals import *
from evaluate import *

def plot_history(history, model_folder_path = 'saved_models',show_histograms = False, show_boxplots = False,
                    show_kde = False, filter_outliers = True, save = True):

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
    initialise_vars(history, model_folder_path)

    #plot train and validation accuracy on history
    plt.figure()
    plt.plot(history_accuracy_array)
    plt.plot(history_val_accuracy_array)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.grid()
    if save:
        plt.savefig((os.path.join(plots_path, accuracy_fig_filename)), dpi=200)
        # plt.savefig((plots_path + accuracy_fig_filename), dpi=200)
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
    if save:
        plt.savefig((os.path.join(plots_path, loss_fig_filename)), dpi=200)
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
    if save:
        plt.savefig((os.path.join(plots_path, mae_fig_filename)), dpi=200)
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
    if save:
        plt.savefig((os.path.join(plots_path, mse_fig_filename)), dpi=200)
    plt.close()

    #plot train and validation Recall
    plt.figure()
    plt.plot(history_recall_array)
    plt.plot(history_val_recall_array)
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train_recall', 'val_recall'], loc='upper left')
    plt.grid()
    if save:
        plt.savefig((os.path.join(plots_path, recall_fig_filename)), dpi=200)
    plt.close()

    #plot train and validation Precision
    plt.figure()
    plt.plot(history_precision_array)
    plt.plot(history_val_precision_array)
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train_precision', 'val_precision'], loc='upper left')
    plt.grid()
    if save:
        plt.savefig((os.path.join(plots_path, precision_fig_filename)), dpi=200)
    plt.close()

    if (show_histograms):
        plot_histograms(history, save, filter_outliers)

    if (show_boxplots):
        plot_boxplots(history, save, filter_outliers)

    if (show_kde):
        plot_kde(history, save, filter_outliers)

    #plot learning rate during training of model
    plot_lr(history)

def plot_boxplots(history, save, filter_outliers):

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
    if save:
        plt.savefig((os.path.join(plots_path, accuracy_box_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, loss_box_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, mse_box_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, mae_box_filename)), dpi=200)
    plt.close()

def plot_histograms(history, save, filter_outliers):

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
    if save:
        plt.savefig((os.path.join(plots_path, accuracy_hist_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, loss_hist_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, mae_hist_filename)), dpi=200)
    plt.close()

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
    if save:
        plt.savefig((os.path.join(plots_path, mse_hist_filename)), dpi=200)
    plt.close()

def plot_kde(history, save, filter_outliers):

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
    if save:
        plt.savefig((os.path.join(plots_path, accuracy_kde_filename)), dpi=200)
    plt.close()

    #Loss KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_loss_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_loss_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Loss', fontsize = 20)
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.xlabel("Loss", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    if save:
        plt.savefig((os.path.join(plots_path, loss_kde_filename)), dpi=200)
    plt.close()

    #Mean Absolute Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mae_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mae_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Absolute Error', fontsize = 17)
    plt.legend(['train_mae', 'val_mae'], loc='upper right')
    plt.xlabel("Mean Absolute Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    if save:
        plt.savefig((os.path.join(plots_path, mae_kde_filename)), dpi=200)
    plt.close()

    #Mean Squared Error KDE
    plt.figure(figsize=(10,8), dpi= 200)
    sns.kdeplot(history_mse_array, shade=True, color="b", label="loss", alpha=.5)
    sns.kdeplot(history_val_mse_array, shade=True, color="g", label="val_loss", alpha=.5)
    plt.title('KDE Plot for Training/Validation Mean Squared Error', fontsize = 17)
    plt.legend(['train_mse', 'train_mse'], loc='upper right')
    plt.xlabel("Mean Squared Error", fontsize = 15)
    plt.ylabel("Kernel Density Estimate", fontsize = 15)
    plt.grid()
    if save:
        plt.savefig((os.path.join(plots_path, mse_kde_filename)), dpi=200)
    plt.close()

def plot_lr(history):

    """
    Description:
        Plotting and visualising the learning rate during training of model.
    Args:
        history (dict): model training history
    Returns:
        None
    """
    learning_rate = history['lr']
    epochs = range(1, len(learning_rate) + 1)
    plt.plot(epochs, learning_rate)
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.grid()
    plt.savefig((os.path.join(plots_path, 'lr_epochs.png')), dpi=200)
    plt.close()


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
    protein_labels_filename = 'protein_labels_' + calling_test_dataset + '.png'

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
    plt.savefig((os.path.join(plots_path, protein_labels_filename)), dpi=200)
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

def initialise_vars(history,model_folder_path):

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
    plots_path = os.path.join(model_folder_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    #convert history into a dataframe
    history_df = pd.DataFrame(history.items(), columns =['Metrics','Score'], index = history.keys())
    del history_df['Metrics']
    history_df_trans = history_df.T     #transpose dataframe

    ### initialise global numpy arrays used for storing metrics from model history ###

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

if __name__ == '__main__':

    #get input arguments
    parser = argparse.ArgumentParser(description='Plotting visualisations for model')
    parser.add_argument('-model_folder', '--model_folder', type=str.lower, required=True ,default='../saved_models',
                        help='Name of model folder to plot')
    parser.add_argument('-history_path', '--history_path', type=str.lower, required = True,
                        help='Filepath for history')
    parser.add_argument('-show_hist', '--show_hist', type=bool, required = False, default = False,
                        help='Plot histograms of metrics from model - default is False')
    parser.add_argument('-show_box', '--show_box', type=str.lower, required = False, default = False,
                        help='Plot Boxplots of metrics from model - default is False')
    parser.add_argument('-show_kde', '--show_kde', type=str.lower, required = True, default = False,
                        help='Plot Kernel Density Estimates of metrics from model - default is False')
    parser.add_argument('-save', '--save', type=bool, required = True, default = False,
                        help='Select whether to save the plots after showing, default is False')

    #parse input arguments
    args = parser.parse_args()
    history_path = args.history_path
    model_folder_path = args.model_folder
    show_hist = args.show_hist
    show_box = args.show_box
    show_kde = args.show_kde
    save = args.save

    #path to model folder must exist
    if not (os.path.isdir(os.path.join(os.path.dirname(os.getcwd()),model_folder_path))):
        print('Model Folder path of model to plot does not exist')
        sys.exit(0)

    #open pickle of history
    try:
        f = open(history_path, 'rb')
        history = pickle.load(f)
    except IOError:
        print('Error opening file')
    except pickle.UnpicklingError as e:
        print(traceback.format_exc(e))
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        sys.exit(0)
    f.close()
