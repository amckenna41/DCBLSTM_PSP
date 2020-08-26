import time
#Importing libraries and dependancies required for building the model
import numpy as np
import gzip
import h5py
import tensorflow as tf
import argparse
import random as rn
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io
import os
import sys
import importlib
from datetime import date
from datetime import datetime
import hypertune
from google.cloud import storage, exceptions
import subprocess
import pickle
import json
from training.get_dataset import *
from training.plot_model import *

# storage_client = storage.Client.from_service_account_json("service-account.json")
# storage_client = storage.Client.from_service_account_json("psp-keras-training.json")
# storage_client = storage.Client.from_service_account_json("service-account.json")
BUCKET_NAME = "keras-python-models"
storage_client = storage.Client()
# storage_client = storage.Client.from_service_account_json("psp-keras-training.json")
bucket = storage_client.get_bucket(BUCKET_NAME)

def upload_history(history, model_save_path, score):

    print('set gcp env var')
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


    blob_path = 'history/history_'+ str(datetime.date(datetime.now())) + \
        '_' + str((datetime.now().strftime('%H:%M'))) +'.pckl'
    blob = bucket.blob(blob_path)
    upload_file(blob_path,history_filepath)
    time.sleep(2)

    #
    # blob.download_to_filename(history_filepath)
    # time.sleep(2)
    # upload_file(blob_path,history_filepath)

    # time.sleep(2)
    # blob.upload_from_filename(history_filepath)

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
#     metadata['Model_Name'] = model_save_path
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


def upload_model(model, args, model_save_path):

    print('Saving model')
    # #get model name by accessing args
    # if (args.bidirection):
    #     model_prefix = 'model_blstm_hpconfig_'
    # else:
    #     model_prefix = 'model_lstm_hpconfig_'
    #
    # model_save_path = model_prefix + str(datetime.date(datetime.now())) + \
    #     '_' + str((datetime.now().strftime('%H:%M')))+ '.h5'

    # blob_path = 'models/'+ model_prefix + str(datetime.date(datetime.now())) +\
    #      '_' + str((datetime.now().strftime('%H:%M')))+'.h5'
    blob_path = 'models/' + model_save_path

    model.save(model_save_path)

    # blob = bucket.blob(blob_path)
    # blob.upload_from_filename(model_save_path)
    upload_file(blob_path, model_save_path)


def upload_file(blob_path, filepath):

    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(filepath)

    #blob_path is GCP Storage filepath
    #filepath is local path to file

def download_file(blob_path, filepath):

    print('Downloading file...')
    blob = bucket.blob(blob_path)
    blob.download_to_filename(filepath)

def list_bucket_objects(bucket_name):
    # """Lists all the blobs in the bucket."""
    # # bucket_name = "your-bucket-name"
    #
    # storage_client = storage.Client()
    #
    # # Note: Client.list_blobs requires at least package version 1.17.0.
    # blobs = storage_client.list_blobs(bucket_name)
    #
    # for blob in blobs:
    #     print(blob.name)
    pass

def delete_blob(bucket_name, blob_name):
    #     """Deletes a blob from the bucket."""
    # # bucket_name = "your-bucket-name"
    # # blob_name = "your-object-name"
    #
    # storage_client = storage.Client()
    #
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(blob_name)
    # blob.delete()
    #
    # print("Blob {} deleted.".format(blob_name))
    pass

#
# def view_bucket_iam_members(bucket_name):
#     """View IAM Policy for a bucket"""
#     # bucket_name = "your-bucket-name"
#
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#
#     policy = bucket.get_iam_policy(requested_policy_version=3)
#
#     for binding in policy.bindings:
#         print("Role: {}, Members: {}".format(binding["role"], binding["members"]))

#update iam policy of bucket so above functions can work
def update_bucket_policy(bucket_name):
    pass
