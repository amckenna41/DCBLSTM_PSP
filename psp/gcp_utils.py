
################################################################################
########                Google Cloud Platform Utilities                 ########
################################################################################

#Importing required libraries and dependancies
import numpy as np
import pandas as pd
import os
import glob
import subprocess
from google.cloud import storage, exceptions
from googleapiclient import errors
from googleapiclient import discovery
# from google.cloud import logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
from tensorflow.keras.utils import plot_model
import pickle
import json
from subprocess import Popen, PIPE
try:
    from _globals import *
except:
    from . _globals import *

#initialise storage bucket object
BUCKET = None

def initialise_bucket(bucket):
    """
    Description:
        Initialise GCP storage bucket and client.
    Args:
        :bucket (str): name of GCP Storage bucket.
    Returns:
        None
    """
    global BUCKET
    storage_client = storage.Client()
    bucketName = bucket.replace('gs://','')
    bucket = storage_client.get_bucket(bucketName)
    BUCKET = bucket

def save_history(history, model_output_folder):
    """
    Description:
        Save and upload model history to GCP Storage.
    Args:
        :history (dict): model history.
        :model_output_folder (str): output folder where all models assets and results are stored.
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
        print('Error creating history pickle.')

    #create history blob path
    blob_path = os.path.join(model_output_folder, 'history.pckl')

    #upload history to bucket
    upload_file(blob_path,'history.pckl')

def upload_directory(local_path, gcs_folder_path):
    """
    Description:
        Upload directory recursively to GCP Storage.
    Args:
        :local_path (str): local path to directory.
        :gcs_folder_path (str): blob folder path on GCP.
    Returns:
        None
    Reference:
        https://stackoverflow.com/questions/25599503/how-to-upload-folder-on-google-cloud-storage-using-python-api
    """
    if not (os.path.isdir(local_path)):
        raise OSError('Path to local directory not found.')
        return

    #recursively iterate through directory, uploading each file individually to GCP bucket.
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
           upload_directory(local_file, gcs_folder_path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(gcs_folder_path, local_file[1 + len(local_path):])
           upload_file(remote_path, local_file)

def upload_file(blob_path, filepath):
    """
    Description:
        Upload blob to bucket.
    Args:
        :blob_path (str): path of blob object in bucket.
        :filepath (str): local filepath of object.
    Returns:
        None
    """
    #initialise blob in bucket
    blob = BUCKET.blob(blob_path)

    #upload blob to specified bucket
    try:
        blob.upload_from_filename(filepath)
    except Exception as e:
        print("Error uploading blob {} to storage bucket {} ".format(blob_path, e.message))

def download_file(blob_path, filepath):
    """
    Description:
        Download blob object from GCP Storage bucket to local dir.
    Args:
        :blob_path (str): path of blob object in bucket.
        :filepath (str): local filepath for downloaded blob.
    Returns:
        None
    """
    #initialise blob in bucket
    blob = BUCKET.blob(blob_path)

    #download blob from GCP Storage bucket to local filepath
    try:
        blob.download_to_filename(filepath)
        print('Blob {} downloaded to {}.'.format(blob_path, filepath))
    except  Exception as e:
        print("Error downloading blob {} from storage bucket {} ".format(blob_path, e.message))

def get_job_hyperparmeters(project_id, job_name):
    """
    Description:
        Output results from hyperparameter tuning process to csv.
    Args:
        :project_id (str): name of GCP project.
        :job_name (str): name of GCP Ai-Platform job.
    Returns:
        :df (pandas DataFrame): dataframe of hyperparameters and their associated results from training.
    """
    #get GCP credentials for making request
    job_id = '{}/jobs/{}'.format(project_id, job_name)
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)

    #make request to hyperparameter job using Google API Client
    try:
        request = ml.projects().jobs().get(name=job_id).execute()
        if request['state'] != "SUCCEEDED":     #check that job has completed
            print('Hyperparameter tuning job not completed.')
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
        List all objects in bucket.
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

def delete_blob(blob_name):
    """
    Description:
        Delete blob from GCP storage bucket.
    Args:
        :blob_name (str): name of blob to delete.
    Returns:
        None
    """
    #get blob from bucket using GCP storage API
    blob = BUCKET.blob(blob_name)

    #delete blob
    try:
        blob.delete()
        print("Blob {} deleted from {} bucket".format(blob_name, BUCKET_NAME))
    except Exception as e:
        print("Error deleting blob {} from {} bucket: {}".format(blob_name, BUCKET_NAME, e.message))

def get_model_output(job_name):
    """
    Description:
        Output results from metrics captured when training model. Output to CSV.
    Args:
        :job_name (str): name of AI Platform training job
    Returns:
        None
    """
    #local filepath and GCP blob name
    model_output_csv_blob = os.path.join(job_name, 'output_results.csv')

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])

    #exporting Dataframe to CSV
    model_output_df.to_csv('output_results.csv',index=False)

    #uploading blob to cloud storage
    upload_file(os.path.join(job_name, 'output_results.csv'),'output_results.csv')

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

def append_all_output(output_results_df, all_results="all_results.csv"):
    """
    Description:
        Append training results/parameters of current job to CSV containing
        results/parameters of all previous jobs.
    Args:
        :all_results (str): filepath to csv containing results of previous jobs.
        :all_results (str): filepath to csv containing results of current job.
    Returns:
        None
    """
    #check if results file exists in bucket, if so then download locally
    if (BUCKET.blob(all_results).exists()):
        download_file(all_results, all_results)
    else:
        return

    #read csv results file
    all_results_df = pd.read_csv(all_results)
    #append results of current training job to all_results file
    all_results_df = all_results_df.append(output_results_df)
    #export results to csv
    all_results_df.to_csv(all_results, index=False)

    #upload updated results file to bucket
    upload_file(all_results, all_results)

def parse_json_arch(arch_json):
    """
    Description:
        Parse model architecture JSON.
    Args:
        :arch_json (str): filepath to model json
    Returns:
        None
    """
    pass

def get_job_status(job_name):
    """
    Description:
        Get training status of GCP Ai-Platform training job.
    Args:
        :job_name (str): name of training job.
    Returns:
        :status (str): training status of job.
        :err_message (str): error message of job.
    """
    # job_logs = subprocess.check_output(["gcloud", "ai-platform","jobs","describe",job_name])
    job_logs = subprocess.Popen(["gcloud", "ai-platform","jobs","describe",job_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)

    output, err = job_logs.communicate(b"input data that is passed to subprocess' stdin")

    #parse job status from command-line output
    status=""
    for item in (output.decode('UTF-8')).split("\n"):
        if ("state:" in item):
            status = item.strip()
            status = (status[status.find(':')+1:]).strip()

    #parse error message from command-line output
    err_message=""
    if (status=="FAILED"):
        for item in (output.decode('UTF-8')).split("\n"):
            if ("errorMessage:" in item):
                err_message = item.strip()

    #get err_message down to etag
    return status, err_message

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
    def __init__(self, initAlpha=0.01, epochs=100):
        self.initAlpha = initAlpha
        self.epochs = epochs
        decay = self.initAlpha / self.epochs

    def __call__(self, lr, epochs):
        decay = self.initAlpha / self.epochs
        return ((lr *1) / (1 + decay * self.epochs))
