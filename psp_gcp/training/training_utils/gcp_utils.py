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
import pandas as pd
import pickle
import json
from training.training_utils.get_dataset import *
from training.training_utils.plot_model import *
# storage_client = storage.Client.from_service_account_json("service-account.json")

#initialise bucket name and GCP storage client
global BUCKET_NAME
BUCKET_NAME = "keras-python-models-2"
storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
#credentials = GoogleCredentials.get_application_default()

#save and upload model history to bucket
def upload_history(history, score, model_blob_path):

    # storage_client = storage.Client()
    # # storage_client = storage.Client.from_service_account_json("psp-keras-training.json")
    # bucket = storage_client.get_bucket("keras-python-models")
    # buckets = list(storage_client.list_buckets())
    # print(buckets)
    #Saving pickle of history so that it can later be used for visualisation of the model
    history_filepath = 'history_' + str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) +'.pckl'

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

    # blob_path = 'history/history_'+ str(datetime.date(datetime.now())) + \
    #     '_' + str((datetime.now().strftime('%H:%M'))) +'.pckl'
    blob_path = str(model_blob_path) + 'history/history_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) +'.pckl'

    blob = bucket.blob(blob_path)
    upload_file(blob_path,history_filepath)
    time.sleep(2)

    ## Set MetaData of history blob to store results from history ##

#     history_meta = {}
#     for key, value in (history.history.items()):
#         if 'val_false' in key or 'false' in key:
#             # history_meta[key] = ([float(i) for i in([ '%.1f' % elem for elem in history.history[key]])])
#             history_meta[key] = ([ '%.1f' % elem for elem in history.history[key]])
#
#         else:
#             # history_meta[key] = ([float(i) for i in([ '%.4f' % elem for elem in history.history[key]])])
#             history_meta[key] = ([ '%.4f' % elem for elem in history.history[key]])
#
#     time.sleep(2)
#
#
#     metadata = history_meta
#     metadata['best_accuracy'] = max(history_meta['accuracy'])
#     metadata['best_val_accuracy'] = max(history_meta['val_accuracy'])
#     metadata['best_loss'] = min(history_meta['loss'])
#     metadata['best_val_loss'] = min(history_meta['val_loss'])
#     metadata['best_mean_squared_error'] = min(history_meta['mean_squared_error'])
#     metadata['best_val_mean_squared_error'] = min(history_meta['val_mean_squared_error'])
#     metadata['best_false_negatives'] = min(history_meta['false_negatives'])
#     metadata['best_false_positives'] = min(history_meta['false_positives'])
#     metadata['best_val_false_negatives'] = min(history_meta['val_false_negatives'])
#     metadata['best_val_false_positives'] = min(history_meta['val_false_positives'])
#     metadata['best_mean_absolute_error'] = min(history_meta['mean_absolute_error'])
#     metadata['best_val_mean_absolute_error'] = min(history_meta['val_mean_absolute_error'])
#     metadata['Evaluation_Loss'] = str(score[0])
#     metadata['Evaluation_Accuracy'] = str(score[1])
#     metadata['Model_Name'] = model_blob_path
#
#     #do statistical analysis/summary stats on above variables e.g std dev, variance,
#     #create json
#     blob.metadata = metadata
#     try:
#         blob.patch()
#     # except exceptions.NotFound:,   except google.api_core.exceptions.Forbidden:
#     except exceptions.Forbidden:
#         raise ValueError("Error: Access to GCP Storage bucket forbidden, check IAM policy, 403 Error")
#     except exceptions.NotFound:
#         raise ValueError("Error: Access to GCP Storage bucket forbidden, check IAM policy, 404 Error")
#     except exceptions.PermissionDenied:
#         raise ValueError("Error: Access to GCP Storage bucket forbidden, check IAM policy")
#     except exceptions.TooManyRequests:
#         raise ValueError("Error: Access to GCP Storage bucket forbidden, check IAM policy")
#     #https://googleapis.dev/python/google-api-core/latest/exceptions.html
#         # call get_iam_policy and change_iam_policy func to view and change IAM policy to get rid of error
# #cloudstorage.Error, cloudstorage.AuthorizationError, cloudstorage.ForbiddenError, cloudstorage.NotFoundError, cloudstorage.TimeoutError

#save and upload model to bucket
def upload_model(model, model_blob_path,model_save_path):

    #model.get_layer dense_1
    #get file name from args
    print('Saving model')

    model.save(model_save_path)
    upload_file(model_blob_path, model_save_path)

#upload blob to bucket
def upload_file(blob_path, filepath):

    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(filepath)

    #blob_path is GCP Storage filepath
    #filepath is local path to file

#download blob from bucket to local dir
def download_file(blob_path, filepath):

    print('Downloading file...')
    blob = bucket.blob(blob_path)
    blob.download_to_filename(filepath)

def get_best_model(project_id, job_name):

    # Define the credentials for the service account
    credentials = service_account.Credentials.from_service_account_file("service-account.json")
    #credentials = GoogleCredentials.get_application_default()

    project_id = 'projects/{}'.format(project_id)
    job_id = '{}/jobs/{}'.format(project_id, job_name)

    ml = discovery.build('ml', 'v1', credentials=credentials)

    try:
        request = ml.projects().jobs().get(name=job_id).execute()
    except errors.HttpError as err:
        print('Error getting job details')
        print(err._get_reason())

    #get first best model
    best_model = request['trainingOutput']['trials'][0]

    print('Best Hyperparameters:')
    print(json.dumps(best_model, indent=4))

# Create a list for each field

    trial_id, eval_score, conv1_filters, conv1_filters, conv3_filters, window_size,  conv2d_dropout, \
    kernel_regularizer, pool_size, recurrent_layer1, recurrent_layer1, recurrent_dropout, \
    recurrent_recurrent_dropout, after_recurrent_dropout, bidirection, recurrent_layer, \
    dense_1, dense_2, dense_3, dense_4, dense_dropout, optimizer, learning_rate, epochs, \
    batch_size, elapsed_time = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
    [], [], [], [], [], [], [], [], [], [], []

    # Loop through the json and append the values of each field to the lists
    for each in request['trainingOutput']['trials']:
        trial_id.append(each['trialId'])
        eval_score.append(each['finalMetric']['eval_score'])
        conv1_filters.append(each['hyperparameters']['conv1_filters'])
        conv2_filters.append(each['hyperparameters']['conv2_filters'])
        conv3_filters.append(each['hyperparameters']['conv3_filters'])
        window_size.append(each['hyperparameters']['window_size'])
        conv2d_dropout.append(each['hyperparameters']['conv2d_dropout'])

    # Put the lsits into a df, transpose and name the columns
    df = pd.DataFrame([trial_id, eval_score, conv1_filters, conv2_filters, conv3_filters, window_size, conv2d_dropout]).T
    df.columns = ['trial_id', 'eval_score', 'conv1_filters', 'conv2_filters', 'conv3_filters', 'window_size', 'conv2d_dropout']

    # Display the df
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


# """View IAM Policy for a bucket"""
def view_bucket_iam_members():

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    for binding in policy.bindings:
        print("Role: {}, Members: {}".format(binding["role"], binding["members"]))

#update iam policy of bucket so above functions can work
def update_bucket_policy(bucket_name):
    pass
