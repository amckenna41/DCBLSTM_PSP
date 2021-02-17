
#########################################################################
###                  Google Cloud Platform Utilities                  ###
#########################################################################

#Importing required libraries and dependancies
import numpy as np
import pandas as pd
import os
import time
from google.cloud import storage, exceptions
from googleapiclient import errors
from googleapiclient import discovery
from google.cloud import logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
from training.training_utils.globals import *
import pickle
import json

def save_history(history, job_name):

    """
    Description:
        Save and upload model history to GCP Storage

    Args:
        history (dict): model history
        job_name (str): name of GCP Ai-Platform job

    Returns:
        None
    """

    #open history pickle file for writing
    try:
        f = open('history.pckl', 'wb')
        pickle.dump(history.history, f)
        f.close()
    except pickle.UnpicklingError as e:
        print('Error', e)
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        print('Error creating history pickle')

    #upload history blob
    blob_path = os.path.join(job_name, 'history.pckl')
    blob = bucket.blob(blob_path)

    #upload history to bucket
    upload_file(blob_path,'history.pckl')


def upload_file(blob_path, filepath):

    """
    Description:
        Upload blob to bucket
    Args:
        blob_path (str): path of blob object in bucket
        filepath (str): local filepath of object

    Returns:
        None
    """

    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)

    try:
        blob.upload_from_filename(filepath)
    except Exception as e:
        print("Error uploading blob {} to storage bucket {} ".format(blob_path, e.message))

def download_file(blob_path, filepath):

    """
    Description:
        Download blob object from GCP Storage bucket to local dir
    Args:
        blob_path (str): path of blob object in bucket
        filepath (str): local filepath for downloaded blob
    Returns:
        None
    """

    blob = bucket.blob(blob_path)

    try:
        blob.download_to_filename(filepath)
        print('Blob {} downloaded to {}.'.format(blob_path, filepath))
    except  Exception as e:
        print("Error downloading blob {} from storage bucket {} ".format(blob_path, e.message))


def get_job_hyperparmeters(project_id, job_name):

    """
    Description:
        Output results from hyperparameter tuning process to csv
    Args:
        project_id (str): name of GCP project
        job_name (str): name of GCP Ai-Platform job
    Returns:
        df (pandas DataFrame): dataframe of hyperparameters and their associated results from training
    """

    #get GCP credentials for making request
    job_id = '{}/jobs/{}'.format(project_id, job_name)
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)

    #make request to hyperparameter job using Google API Client
    try:
        request = ml.projects().jobs().get(name=job_id).execute()
        if request['state'] != "SUCCEEDED":     #check that job has completed
            print('Hyperparameter tuning job not completed')
            return
    except errors.HttpError as err:
        print('Error getting job details')
        print(err._get_reason())

    col = []
    row = []

    #set the columns of the dataframe to the hyperparameter variables
    for column in request['trainingOutput']['trials'][0]['hyperparameters']:
        col.append(column)

    #for each successful trial, append each hyperparameter metric to the row array
    for cols in col:
        for trial in range(0, len(request['trainingOutput']['trials'])):
          #check hyperparameter has SUCCEEDED
          if request['trainingOutput']['trials'][trial]['state'] == "SUCCEEDED":
             row.append(request['trainingOutput']['trials'][trial]['hyperparameters'][cols])

    #transform row list into a numpy array
    row_np = np.asarray(row)
    num_params = len(request['trainingOutput']['trials'][0]['hyperparameters'])

    #horizontally split numpy array into each of the different columns
    row_np = np.split(row_np, num_params)
    #create dataframe from hyperparameter metrics
    df = pd.DataFrame(row_np)
    #transpose dataframe
    df = df.T
    #set columns of dataframe to metric names
    df.columns = col

    #append evaluation score and trial ID to dataframe
    eval_score = []
    trial_id = []
    for trial in range(0, len(request['trainingOutput']['trials'])):
       eval_score.append(request['trainingOutput']['trials'][trial]['finalMetric']['objectiveValue'])
       trial_id.append(request['trainingOutput']['trials'][trial]['trialId'])

    df['eval_score'] = eval_score
    df['trial_id'] = trial_id
    #sort dataframe by the evaluation score
    df.sort_values('eval_score')

    #put evaluation score and trial ID to beginning of dataframe columns
    eval = df['eval_score']
    df.drop(labels=['eval_score'], axis=1,inplace = True)
    df.insert(0, 'eval_score', eval)

    trial_id = df['trial_id']
    df.drop(labels=['trial_id'], axis=1,inplace = True)
    df.insert(0, 'trial_id', trial_id)

    #export dataframe to a csv
    df_filename = job_name + "_hyperparameter_tuning"
    df.to_csv(df_file_name, encoding='utf-8', index=False)

    #upload csv to GCP Storage
    blob_path = job_name + ".csv"
    upload_file(blob_path,df_filename)

    return df

def list_bucket_objects():

    """
    Description:
        List all objects in bucket
    Args:
        None
    Returns:
        None
    """

    #use Google Storage API to get and print all objects in bucket
    try:
        blobs = storage_client.list_blobs(BUCKET_NAME)  #global BUCKET_NAME variable
    except Exception as e:
        print("Error listing blobs from {} bucket: {}".format(BUCKET_NAME, e.message))

    #print bucket blobs
    for blob in blobs:
        print(blob.name)


def get_model_output(job_name):

    """
    Description:
        Output results from metrics captured when training model. Output to CSV
    Args:
        None
    Returns:
        None
    """
    #local filepath and GCP blob name
    model_output_csv_blob = os.path.join(job_name, 'output_results.csv')

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])
    #transposing model_output Dataframe
    # model_output_df_t = model_output_df.transpose()
    #setting 'values' as dataframe column name
    # model_output_df_t.columns = ['Values']

    #exporting Dataframe to CSV
    model_output_df.to_csv('output_results.csv',index=False)

    #add additonal column that multiples values * 100, except for loss/MAE/MSE etc

    #uploading blob to cloud storage
    upload_file(os.path.join(job_name, 'output_results.csv'),'output_results.csv')


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

def parse_json_arch(arch_json):
    pass
    
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


def delete_blob(blob_name):

    """
    Description:
        Delete blob from GCP storage bucket
    Args:
        blob_name (str): name of blob to delete
    Returns:
        None
    """
    #get blob from bucket using GCP storage API
    blob = bucket.blob(blob_name)

    #delete blob
    try:
        blob.delete()
        print("Blob {} deleted from {} bucket".format(blob_name, BUCKET_NAME))
    except Exception as e:
        print("Error deleting blob {} from {} bucket: {}".format(blob_name, BUCKET_NAME, e.message))
