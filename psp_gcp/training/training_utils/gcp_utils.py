#Importing libraries and dependancies required for building the model
import numpy as np
import pandas as pd
import os
import sys
from datetime import date
import time
from datetime import datetime
from google.cloud import storage, exceptions
from googleapiclient import errors
from googleapiclient import discovery
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
import pickle
import json
# storage_client = storage.Client.from_service_account_json("service-account.json")

#initialise bucket name and GCP storage client
global BUCKET_NAME
BUCKET_NAME = "keras-python-models-2"
global current_datetime
current_datetime = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
#credentials = GoogleCredentials.get_application_default()

#save and upload model history to bucket
def upload_history(history, score, model_blob_path):

    #Saving pickle of history so that it can later be used for visualisation of the model
    history_filepath = 'history_' + current_datetime +'.pckl'

    #open history
    try:
        # f = open(BUCKET_NAME + '/history/history'+ str(datetime.date(datetime.now())) +'.pckl', 'wb')
        f = open(history_filepath, 'wb')
        pickle.dump(history.history, f)
        f.close()
    except pickle.UnpicklingError as e:
        print('Error', e)
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
    except Exception as e:
        print(traceback.format_exc(e))
        print('Error creating history pickle')


    blob_path = str(model_blob_path) + 'history/history_'+ current_datetime +'.pckl'

    #upload history to bucket
    blob = bucket.blob(blob_path)
    upload_file(blob_path,history_filepath)
    time.sleep(2)

    ## Set MetaData of history blob to store results from history ##

    #set metrics to 4 dp except for False Positives metric which is set to 1dp
    history_meta = {}
    for key, value in (history.history.items()):
        if 'val_false' in key or 'false' in key:
            history_meta[key] = ([ '%.1f' % elem for elem in history.history[key]])

        else:
            history_meta[key] = ([ '%.4f' % elem for elem in history.history[key]])

    time.sleep(2)

    #set metadata tags in bucket blob to the values from history
    metadata = history_meta
    metadata['best_accuracy'] = max(history_meta['accuracy'])
    metadata['best_val_accuracy'] = max(history_meta['val_accuracy'])
    metadata['best_loss'] = min(history_meta['loss'])
    metadata['best_val_loss'] = min(history_meta['val_loss'])
    metadata['best_mean_squared_error'] = min(history_meta['mean_squared_error'])
    metadata['best_val_mean_squared_error'] = min(history_meta['val_mean_squared_error'])
    metadata['best_false_negatives'] = min(history_meta['false_negatives'])
    metadata['best_false_positives'] = min(history_meta['false_positives'])
    metadata['best_val_false_negatives'] = min(history_meta['val_false_negatives'])
    metadata['best_val_false_positives'] = min(history_meta['val_false_positives'])
    metadata['best_mean_absolute_error'] = min(history_meta['mean_absolute_error'])
    metadata['best_val_mean_absolute_error'] = min(history_meta['val_mean_absolute_error'])
    metadata['Evaluation_Loss'] = str(score[0])
    metadata['Evaluation_Accuracy'] = str(score[1])
    metadata['Model_Name'] = model_blob_path

    #create json
    blob.metadata = metadata
    try:
        blob.patch()
    # except exceptions.NotFound:,   except google.api_core.exceptions.Forbidden:
    except exceptions.Forbidden:
        raise ValueError("Error: Access to GCP Storage bucket forbidden, check IAM policy, 403 Error")
    except exceptions.NotFound:
        raise ValueError("Error: GCP Storage bucket not found 404 Error")
    except exceptions.PermissionDenied:
        raise ValueError("Error: Access to GCP Storage bucket denied, check IAM policy")
    except exceptions.TooManyRequests:
        raise ValueError("Error: Too many access requests to GCP Storage bucket")
    #https://googleapis.dev/python/google-api-core/latest/exceptions.html
    # call get_iam_policy and change_iam_policy func to view and change IAM policy to get rid of error
    #cloudstorage.Error, cloudstorage.AuthorizationError, cloudstorage.ForbiddenError, cloudstorage.NotFoundError, cloudstorage.TimeoutError

#save and upload model to bucket
def upload_model(model, model_blob_path,model_save_path):

    print('Saving model')

    model.save(model_save_path)
    upload_file(model_blob_path, model_save_path)

#upload blob to bucket
def upload_file(blob_path, filepath):

    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(filepath)

#download blob from bucket to local dir
def download_file(blob_path, filepath):

    print('Downloading file...')
    blob = bucket.blob(blob_path)
    blob.download_to_filename(filepath)

#output hyperparameter tuning results to csv
def get_job_hyperparmeters(project_id, job_name):

    #make request to hyperparameter job
    job_id = '{}/jobs/{}'.format(project_id, job_name)
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)

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
    df_filename = job_name + "_parameters"
    df.to_csv(df_file_name, encoding='utf-8', index=False)

    df.head()

    return df

#List all objects within bucket
def list_bucket_objects():

    blobs = storage_client.list_blobs(BUCKET_NAME)

    for blob in blobs:
        print(blob.name)

#Delete specified blob from bucket
def delete_blob(blob_name):

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))

#get IAM policy for bucket
def view_bucket_iam_members():

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    for binding in policy.bindings:
        print("Role: {}, Members: {}".format(binding["role"], binding["members"]))
