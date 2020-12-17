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
from google.cloud import logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
from training.training_utils.globals import *
import pickle
import json

#save and upload model history to bucket
def upload_history(history, model_blob_path):

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

#upload blob to bucket
def upload_file(blob_path, filepath):

    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)
    try:
        blob.upload_from_filename(filepath)
    except Exception as e:
        print("Error uploading blob {} to storage bucket {} ".format(blob_path, e.message))

#download blob from bucket to local dir
def download_file(blob_path, filepath):

    blob = bucket.blob(blob_path)
    try:
        blob.download_to_filename(filepath)
        print('Blob {} downloaded to {}.'.format(blob_path, filepath))
    except  Exception as e:
        print("Error downloading blob {} from storage bucket {} ".format(blob_path, e.message))


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

    try:
        blobs = storage_client.list_blobs(BUCKET_NAME)
    except Exception as e:
        print("Error listing blobs from {} bucket: {}".format(BUCKET_NAME, e.message))

    for blob in blobs:
        print(blob.name)

#
# class Logging:
#
#     def __init__(self, project_id, job_id, topic_id, sink_id, subscription_id):
#         self.project_id = project_id
#         self.job_id = job_id
#         self.topic_id = topic_id
#         self.sink_id = sink_id
#
#     def list_sinks():
#         pass
      # def create_sinks():
      #     pass
#
#     def update_sink(self, sink_id, filter_):
#         pass
#
#     def create_topic():
#         pass
#
#     def list_topics():
#         pass
#
#     def delete_topic():
#         pass
#
#     def create_subscription():
#         pass
#
#     def delete_subscription():
#         pass
#
#     def list_subscriptions_in_topic():
#         pass
#
#     def receive_messages(project_id, subscription_id, timeout=None):
#         pass
#
#
#
# def create_sink(sink_name, destination_bucket, filter_, project_id, topic_id):
#     logging_client = logging.Client()
#
#
#     destination = "pubsub.googleapis.com/projects/ninth-optics-286313/topics/ai-platform-status-test"
#
#     #  bucket=destination_bucket)
#
#     sink = logging_client.sink(
#         sink_name,
#         filter_,
#         destination)
#
#     if sink.exists():
#         print('Sink {} already exists.'.format(sink.name))
#         return
#
#     sink.create()
#     print('Created sink {}'.format(sink.name))
# # [END logging_create_sink]
#
#
#     pass
#
# #https://googleapis.dev/python/logging/latest/gapic/v2/api.html
# def check_job_status(project_id, job_id):
#
#     #get job status info
#     # resource.type = "ml_job"
#     # resource.labels.task_name = "service" - get basic service info
#     # resource.labels.job_id = "JOB_ID"
#     #textPayload = "Job completed successfully."
#
#     #get other job outputs from model training
#     # resource.type = "ml_job"
#     # resource.labels.task_name="master-replica-0"
#     # jsonPayload.message:"Precision -" OR jsonPayload.message:"Recall -" OR
#
#     #1. Create Topic
#     #2. Create subscription to topic
#     #3. Create sink (destination is topic)
#     #create log sink
#
#     #update sink filter
#     #resource.labels.job_id = job_id
#     # textPayload = "Job failed."/ "Job cancelled."/ "Job completed successfully."
#
#     pass

def get_model_output():

    model_output_csv = "model_output_csv_" + current_datetime +'.csv'
    model_output_csv_blob = 'models/model_output_csv_' +  current_datetime +'.csv'

    #converting model_output dictionary to pandas Dataframe
    model_output_df = pd.DataFrame(model_output, index=[0])
    #transposing model_output Dataframe
    model_output_df_t = model_output_df.transpose()
    model_output_df_t.columns = ['Values']

    #exporting Dataframe to CSV
    model_output_df_t.to_csv(model_output_csv,columns=['Values'])

    #add additonal column that multiples values * 100, except for loss/MAE/MSE etc
    #uploading blob to cloud storage
    upload_file(model_output_csv_blob, model_output_csv)
    #
    # #downloading model_output to local directory
    # download_file(model_output_csv_blob, model_output_csv_blob)


"""
Delete blob from GCP bucket
blob_name: name of blob to delete from bucket
"""
def delete_blob(blob_name):

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    try:
        blob.delete()
        print("Blob {} deleted from {} bucket".format(blob_name, BUCKET_NAME))
    except Exception as e:
        print("Error deleting blob {} from {} bucket: {}".format(blob_name, BUCKET_NAME, e.message))

# class IAM():
    # #get IAM policy for bucket
    # def view_bucket_iam_members():
    #
    #     policy = bucket.get_iam_policy(requested_policy_version=3)
    #
    #     for binding in policy.bindings:
    #         print("Role: {}, Members: {}".format(binding["role"], binding["members"]))
