
import pickle
import pandas as pd
import os
import math
from globals import *
import numpy as np

def save_history(history, save_path):

    """
    Description:
        Save training model history
    Args:
        history (dict): dictionary containing training history of keras model with all captured metrics
        save_path: path to save history to
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
        Output model results to a CSV
    Args:
        model_save_path (str): filepath for model directory to store output csv in
    Returns:
        None
    """
    save_path = os.path.join(model_save_path, 'model_output.csv')

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])

    # #transposing model_output Dataframe
    # model_output_df_t = model_output_df.transpose()
    # #setting 'values' as dataframe column name
    # model_output_df_t.columns = ['Values']

    #exporting Dataframe to CSV
    # model_output_df_t.to_csv(save_path,columns=['Values'])
    model_output_df.to_csv(save_path,index=False)

    print('Model Output file exported and stored in {} '.format(save_path))

def append_model_output(output_key, output_value):

    """
    Description:
        Appending metrics from model training to model_output dictionary

    Args:
        output_key (str): metric name
        output_value (float): value of metric

    Returns:
        None
    """
    model_output[output_key] = output_value

class StepDecay():

    """
    Description:
        Step Decay Learning rate scheduler.

    Args:
        initAlpha (float):
        factor (float):
        dropEvery (int):

    Returns:
        Result from step decay function
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
        initAlpha (float):
        k (float):

    Returns:
        Result from exponential decay function
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
        initAlpha (float):

    Returns:
        Result from timed based decay function
    """
    def __init__(self, initAlpha=0.01):
        self.initAlpha, initAlpha

        epochs = 100
        decay = initial_learning_rate / epochs

    def __call__(self, lr, epochs):

        decay = self.initAlpha / epochs

        return ((lr *1) / (1 + decay * epochs))

class BatchNorm():
    pass
