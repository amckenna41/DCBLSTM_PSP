#!/bin/bash

#Shell script for making all the resources required for deployment
PROJECT_ID='[PROJECT_ID]' #@param {type:"string"}
REGION='[REGION_NAME]' #e.g 'us-central1', 'europe-west1'
BUCKET_NAME='keras-python-models'
TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'
TEST_PATH = 'cb513+profile_split1.npy'
CASP10_PATH = 'casp10.h5'
CASP11_PATH = 'casp11.h5'

# gcloud components update      #update SDK
gcloud auth list
# gcloud auth login
#
# gcloud init
#
# gcloud version
#
# gcloud projects create $PROJECT_ID
# gcloud config set project $PROJECT_ID
#
# #create bucket in given region
# gsutil mb -l $REGION gs://$BUCKET_NAME
#
# #list contents of bucket
# gsutil ls -r gs://$BUCKET_NAME/**
#
# #get size of bucket
# gsutil du -s gs://$BUCKET_NAME

#create data subdirectory in bucke

echo "Copying datasets from data dir to GCP Storage..."

#upload training and test datasets from local directory to bucket
# gsutil cp ../data/*.npy gs://$BUCKET_NAME/data/
# gsutil cp ../data/*.h5 gs://$BUCKET_NAME/data/

#use of gsutil cp command requires the following storage IAM permissions:
# storage.objects.list2 (for the destination bucket)
# storage.objects.get (for the source objects)
# storage.objects.create (for the destination bucket)
# storage.objects.delete3 (for the destination bucket)
#https://cloud.google.com/storage/docs/access-control/iam-gsutil
#get current project IAM policies - gcloud projects get-iam-policy $PROJECT_ID

#test if training and test datasets already in Google Storage, if not
#upload from local data directory to bucket

gsutil -q stat gs://$BUCKET_NAME/data/$TRAIN_PATH
return_value=$?

if [ $return_value != 0 ]; then
    echo "Dataset already exists"
else
    echo "Copying dataset from data dir"
    gsutil cp ../data/$TRAIN_PATH gs://$BUCKET_NAME/data/
fi

gsutil -q stat gs://$BUCKET_NAME/data/$TEST_PATH
return_value=$?

if [ $return_value != 0 ]; then
    echo "Dataset already exists"
else
    echo "Copying dataset from data dir"
    gsutil cp ../data/$TEST_PATH gs://$BUCKET_NAME/data/
fi

gsutil -q stat gs://$BUCKET_NAME/data/$CASP10_PATH
return_value=$?

if [ $return_value != 0 ]; then
    echo "Dataset already exists"
else
    echo "Copying dataset from data dir"
    gsutil cp ../data/$CASP10_PATH gs://$BUCKET_NAME/data/
fi

gsutil -q stat gs://$BUCKET_NAME/data/$CASP11_PATH
return_value=$?

if [ $return_value != 0 ]; then
    echo "Dataset already exists"
else
    echo "Copying dataset from data dir"
    gsutil cp ../data/$CASP11_PATH gs://$BUCKET_NAME/data/
fi



#create folders in bucket
#create checkpoints directory in bucket
#create models directory in bucket
