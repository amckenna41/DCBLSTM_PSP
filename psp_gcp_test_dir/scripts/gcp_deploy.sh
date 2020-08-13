#!/bin/bash

# BATCH_SIZE = batch_size; EPOCHS = epochs;
echo "Running GCP Script to deploy model to GCP"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
# gcloud components update
echo "Checkpoints and models will be stored at directory $JOB_DIR"
echo "Staging bucket is $JOB_DIR"
echo "Region is $REGION"
echo "Job Name is $JOB_NAME"


#GCS location to write checkpoints and export models

# curl https://sdk.cloud.google.com | bash #download gcloud sdk
# gcloud init
#gcloud ai-platform jobs list - get list of most recent jobs on ai-platform and describe

##### gcloud ai-platform jobs submit training JOB46 --package-path ./training --module-name training.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config training/cloudml-gpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
#temp config file deploy
gcloud ai-platform jobs submit training JOB62 \
--package-path training/ \
--module-name training.task \
--staging-bucket gs://keras-python-models \
--region us-central1 \
--config training/temp_gcp_configfile.yaml \
--runtime-version 2.1 \
--python-version 3.7 \
--job-dir gs://keras-python-models \

# gcloud help
# gcloud ml-engine jobs submit training TPU_JOB1 --package-path ./trainer --module-name trainer.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config trainer/cloudml-tpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
# gcloud ai-platform jobs submit training JOB26 --package-path ./training --module-name training.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config training/cloudml-gpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
# gcloud ai-platform jobs submit training TPU_JOB1 --package-path ./trainer --module-name trainer.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config trainer/cloudml-tpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models

#gsutil acl ch -u AllUsers:R gs://yourbucket/** #making dataset publicly accessible.


#create project if doesn't exist
#gcloud alpha billing accounts list
#
# if [ "$#" -lt 3 ]; then
#    echo "Usage:  ./create_projects.sh billingid project-prefix  email1 [email2 [email3 ...]]]"
#    echo "   eg:  ./create_projects.sh 0X0X0X-0X0X0X-0X0X0X learnml-20170106  somebody@gmail.com someother@gmail.com"
#    exit
# fi
#
# ACCOUNT_ID=$1
# shift
# PROJECT_PREFIX=$1
# shift
# EMAILS=$@
#
# gcloud components update
# gcloud components install alpha
#
# for EMAIL in $EMAILS; do
#    PROJECT_ID=$(echo "${PROJECT_PREFIX}-${EMAIL}" | sed 's/@/x/g' | sed 's/\./x/g' | cut -c 1-30)
#    echo "Creating project $PROJECT_ID for $EMAIL ... "
#
#    # create
#    gcloud alpha projects create $PROJECT_ID
#    sleep 2
#
#    # editor
#    rm -f iam.json.*
#    gcloud alpha projects get-iam-policy $PROJECT_ID --format=json > iam.json.orig
#    cat iam.json.orig | sed s'/"bindings": \[/"bindings": \[ \{"members": \["user:'$EMAIL'"\],"role": "roles\/editor"\},/g' > iam.json.new
#    gcloud alpha projects set-iam-policy $PROJECT_ID iam.json.new
#
#    # billing
#    gcloud alpha billing accounts projects link $PROJECT_ID --account-id=$ACCOUNT_ID
#
# done
