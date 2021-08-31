
################################################################################
#############            GCP Notification Function                ##############
################################################################################

#!/bin/bash

#Help Funtion showing script usage
Help()
{
   echo "Bash Script for utilising GCP notification functionality"
   echo ""
   echo "Basic Usage, using default parameters: ./gcp_notification_func "
   echo "Usage: ./gcp_notification_func [--b|--e|--t|--tr|-g|--bu|--sT|--mT|--h]"
   echo ""
   echo "Options:"
   echo "$1 GCP Storage Bucket name"
   echo "$2 GCP PubSub Topic name"
   echo "$3 GCP PubSub Subscription name"
   echo "$4 Path to source directory for notification function"
   echo "$5 Destination email that will be notified"
   echo "$6 Less secure source email where results will be sent from"
   echo "$7 Password for SMTP client for source email"
   echo "$8 GCP Cloud Function Name"
   echo "$9 GCP Cloud Function runtime version"
   echo ""
   exit
}

### Only need to run this script once before calling gcp_training script ###
BUCKET_NAME=$1
TOPIC=$2
SUBSCRIPTION=$3
SOURCE_DIR=$4
TOMAIL=$5
FROMMAIL=$6
EMAIL_PASS=$7
FUNCTION_NAME="notification_func"
RUNTIME_VERSION="python37"

echo ""
echo "Input Arguments"
echo "#########################"
echo 'BUCKET NAME = ' $1
echo 'TOPIC = ' $2
echo 'SUBSCRIPTION = ' $3
echo 'SOURCE_DIR = ' $4
echo 'TOMAIL = ' $5
echo 'FROMMAIL = ' $6
echo 'EMAIL_PASS = ' $7
echo "#########################"
echo ""

#update any gcloud components
# gcloud components update

#listing existing pubsub topics
gcloud pubsub topics list

#delete topic if already exists
gcloud pubsub topics delete $TOPIC  #* may get error if topic doesn't exist but will not halt execution

#create pubsub topic
gcloud pubsub topics create $TOPIC

#delete pubsub subscription if already exists
gcloud pubsub subscriptions delete $SUBSCRIPTION #* may get error if subscription doesn't exist but will not halt execution

#create subscription to topic
gcloud pubsub subscriptions create $SUBSCRIPTION \
   --topic=$TOPIC

#list bucket notifications
gsutil notification list $BUCKET_NAME

#delete bucket notifications
gsutil notification delete $BUCKET_NAME #* may get error if notification doesn't exist but will not halt execution

#delete gcloud function if exists (need to enable Cloud Build API)
gcloud functions delete $FUNCTION_NAME #* may get error if function doesn't exist but will not halt execution

#deploy gcloud function
gcloud functions deploy $FUNCTION_NAME \
    --source $SOURCE_DIR \
    --runtime $RUNTIME_VERSION \
    --trigger-topic $TOPIC \
    --set-env-vars BUCKET=$BUCKET_NAME,EMAIL_USER=$TOMAIL,FROM_MAIL=$FROMMAIL,EMAIL_PASS=$EMAIL_PASS \
    --allow-unauthenticated

    #Shell script for making all the resources required for deployment
    # PROJECT_ID='[PROJECT_ID]' #@param {type:"string"}
    # REGION='[REGION_NAME]' #e.g 'us-central1', 'europe-west1'
    # BUCKET_NAME='keras-python-models'
    # TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy'
    # TEST_PATH = 'cb513+profile_split1.npy'
    # CASP10_PATH = 'casp10.h5'
    # CASP11_PATH = 'casp11.h5'

    # gcloud components update      #update SDK
    # gcloud auth list
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
