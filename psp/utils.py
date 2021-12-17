################################################################################
############                     Model Utilities                   #############
################################################################################

import pickle
import pandas as pd
import os
import math
try:
    from _globals import *
except:
    from . _globals import *
import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.utils import plot_model

def save_history(history, save_path):
    """
    Description:
        Save training model history.
    Args:
        :history (dict): dictionary containing training history of keras model with all captured metrics.
        :save_path: path to save history to.
    Returns:
        None
    """
    #open history pickle file for writing
    try:
        f = open(save_path, 'wb')
        pickle.dump(history.history, f)
        f.close()
    except pickle.UnpicklingError as e:
        print('Error', e)
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        print('Error creating history pickle')

def get_model_output(model_save_path):
    """
    Description:
        Output model results to a CSV.
    Args:
        :model_save_path (str): filepath for model directory to store output csv in.
    Returns:
        None
    """
    save_path = os.path.join(model_save_path, 'model_output.csv')

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])

    #exporting Dataframe to CSV
    model_output_df.to_csv(save_path,index=False)

    return model_output_df

def visualise_model(model, save_folder):
    """
    Description:
        Visualise Keras TF model, including its layers, connections and data types.
    Args:
        :model (Keras.model): Keras model to visualise.
        :save_folder (str): filepath for model directory to store model img in.
    Returns:
        None
    """
    plot_model(model, to_file=os.path.join(save_folder,'model.png'),
        show_shapes=True, show_dtype=True)

def get_trainable_parameters(model):
    """
    Description:
        Calculate the number of trainable and non-trainable parameters in Keras model.
    Args:
        :model (Keras.model): Keras model to calculate parameters for.
    Returns:
        :trainable_params (int): number of trainable parameters.
        :non_trainable_params (int): number of non-trainable parameters.
        :total_params (int): total number of trainable + non-trainable parameters.
    """
    trainable_params = count_params(model.trainable_weights)
    non_trainable_params = count_params(model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params

    return trainable_params, non_trainable_params, total_params

class StepDecay():
    """
    Description:
        Step Decay Learning rate scheduler.
    Args:
        :initAlpha (float): initial learning rate (default=0.0005).
        :factor (float): drop factor (default=0.8).
        :dropEvery (int): number of epochs learning rate is dropped (default=40).
    Returns:
        Result from step decay function.
    """
    def __init__(self, initAlpha=0.0005, factor=0.8, dropEvery=40):
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        exp = np.floor((epoch + 1) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        return float(alpha)

class ExponentialDecay():
    """
    Description:
        Exponential Decay Learning rate scheduler.
    Args:
        :initAlpha (float): initial learning rate (default=0.0005).
        :k (float): power/exponent of the exponential (default=0.8).
    Returns:
        Result from exponential decay function.
    """
    def __init__(self, initAlpha=0.0005, k=0.8):
        self.initAlpha = initAlpha
        self.k = k

    def __call__(self, epoch):
        return (self.initAlpha * math.exp(-k*epoch))

class TimedBased():
    """
    Description:
        Timed based Decay Learning rate scheduler.
    Args:
        :initAlpha (float): initial learning rate (default=0.0005).
        :epochs (int): number of epochs.
    Returns:
        Result from timed based decay function.
    """
    def __init__(self, initAlpha=0.01):
        self.initAlpha, initAlpha
        epochs = 100
        decay = initial_learning_rate / epochs

    def __call__(self, lr, epochs):
        decay = self.initAlpha / epochs
        return ((lr *1) / (1 + decay * epochs))
