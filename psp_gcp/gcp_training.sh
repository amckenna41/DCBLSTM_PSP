
################################################################################
###############             GCP Training Script                 ################
################################################################################

#!/bin/bash

### check current version of pip and update, if neccessry ###
# python3 -m pip install --user --upgrade pip

#update Python Path
# export PATH="${PATH}:/root/.local/bin"
# export PYTHONPATH="${PYTHONPATH}:/root/.local/bin"

#Help Funtion showing script usage
Help()
{
   echo "Bash Script for building and training PSP model's on GCP"
   echo ""
   echo "Basic Usage, using default parameters: ./gcp_training "
   echo "Usage: ./gcp_training [--b|--e|--t|--tr|-g|--bu|--sT|--mT|--h]"
   echo ""
   echo "Options:"
   echo "-b     batch size for training - default: 256"
   echo "-e     number of epochs to train for - default: 10"
   echo "-td    test dataset to use for evaluation - CB513, CASP10, CASP11, All, default: all"
   echo "-tr    training dataset to use for training - 6133, 5926, default: 5926"
   echo "-g     use GPU for training - 0,1, default: 0 "
   echo "-bu    GCP cloud bucket name to use for storing trained models and all model utilities"
   echo "-sT    scale Tier - default: CUSTOM"
   echo "-mT    master Tier - default: n1-highmem-8"
   echo "-h     help"
   exit
}

for i in "$@"
do
case $i in
    -c=*|--config=*)
    CONFIG="${i#*=}"
    shift # past argument=value
    ;;
    -h|--h)
    Help
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 $1
fi

if [ -z "$CONFIG" ]; then
  CONFIG="config/dummy_config.json"
fi

echo "Config---- $CONFIG"
# MODEL=$(jq .gcp_parameters[0].module_name $CONFIG | tr -d '"')

PACKAGE_PATH=$(jq -r .gcp_parameters[0].package_path $CONFIG)
echo "Package $PACKAGE_PATH"
MODEL="$PACKAGE_PATH."$(jq -r .gcp_parameters[0].module_name $CONFIG)
echo "$MODEL"
JOB_NAME="psp_""$(date +"%Y%m%d_%H%M")"
echo "$JOB_NAME"

STAGING_BUCKET=$(jq -r .gcp_parameters[0].staging_bucket $CONFIG)
echo "$STAGING_BUCKET"
RUNTIME_VERSION=$(jq -r .gcp_parameters[0].runtime_version $CONFIG)
PYTHON_VERSION=$(jq -r .gcp_parameters[0].python_verion $CONFIG)

JOB_DIR=$(jq -r .gcp_parameters[0].job_dir $CONFIG)
echo "jbo-dir$JOB_DIR"
REGION=$(jq -r .gcp_parameters[0].region $CONFIG)
SCALE_TIER=$(jq -r .gcp_parameters[0].scale_tier $CONFIG)
MACHINE_TYPE=$(jq -r .gcp_parameters[0].master_machine_type $CONFIG)

export CUDA_VISIBLE_DEVICES=0   # - initialise CUDA env var
# # CUDA_VISIBLE_DEVICES=1 - If using 1 CUDA enabled GPU


echo "Running model on Google Cloud Platform"
echo ""
echo "Job Details..."
echo "Job Name: $JOB_NAME"
echo "Cloud Runtime Version: $RUNTIME_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "Region: $REGION"
echo "Logs and models stored in bucket: $JOB_DIR"
echo ""

#  submitting Tensorflow training job to Google Cloud
 gcloud ai-platform jobs submit training $JOB_NAME \
     --package-path $PACKAGE_PATH \
     --module-name $MODEL \
     --staging-bucket $STAGING_BUCKET \
     --runtime-version $RUNTIME_VERSION \
     --python-version $PYTHON_VERSION  \
     --job-dir $JOB_DIR \
     --region $REGION \
     --scale-tier $SCALE_TIER \
     --master-machine-type $MACHINE_TYPE \
     -- \
     --config_ $CONFIG


#stream logs to terminal
gcloud ai-platform jobs stream-logs $JOB_NAME


# echo ""
# echo "To view model progress through tensorboard in Google Cloud shell or terminal execute..."
# echo "tensorboard --logdir=$LOGS_DIR --port=8080"
# echo "If in cloud shell, then click on the web preview option "
#
# ##############################################
#
# ###       Results Notification Function    ###
# FUNCTION_NAME="notification_func"
# SOURCE_DIR="notification_func"
# FUNC_VERSION="python37"
# BUCKET_FOLDER="$JOB_NAME/output_results.csv"
# TOPIC="notification_topic"
#
# # # #deploy gcloud function
# gcloud functions deploy $FUNCTION_NAME \
#     --source $SOURCE_DIR \
#     --runtime $FUNC_VERSION \
#     --trigger-topic $TOPIC \
#     --update-env-vars JOB_NAME=$JOB_NAME \
#     --allow-unauthenticated
#
# #create bucket notification to trigger function when output csv lands in bucket folder
# gsutil notification create \
#    -p $BUCKET_FOLDER \
#    -t $TOPIC \
#    -f json \
#    -e OBJECT_FINALIZE $BUCKET_NAME




##Function to get list of available models/input argument parameters

##############################################

### Visualise model results on TensorBoard ###
     #tensorboard --logdir [LOGS_PATH] - path declared in Tensorboard callback:

### Common Errors when running gcloud command, see below link ###
      #https://stackoverflow.com/questions/31037279/gcloud-command-not-found-while-installing-google-cloud-sdk

      #Tensorflow GPU warnings when training job, fix by installing TF GPU dependancies:
      #https://stackoverflow.com/questions/60368298/could-not-load-dynamic-library-libnvinfer-so-6

      #Error viewing Tensorboard from Ai-Platform job page:
      # https://stackoverflow.com/questions/43711110/google-cloud-platform-access-tensorboard

### Other Job Functions ###
      ### Run Model Locally ###
      # echo "Running  model locally..."
      # gcloud config set ml_engine/local_python $(which python3)
      #
      # gcloud ai-platform local train \
      #   --module-name $MODULE \
      #   --package-path $PACKAGE_PATH \
      #   --job-dir $JOB_DIR \
      #   -- \
      #   --epochs $EPOCHS \
      #   --batch_size $BATCH_SIZE \
      #   --logs_dir $LOGS_DIR

      #To cancel current job:
      #gcloud ai-platform jobs cancel $JOB_NAME

      #create ai-platform model
      # gsutil cp -r $LOCAL_BINARIES $REMOTE_BINARIES
      # gcloud ml-engine versions create $VERSION_NAME \
      #                                  --model $MODEL_NAME \
      #                                  --origin $REMOTE_BINARIES \
      #                                  --runtime-version 1.10

      #verify model exists
      # gcloud ml-engine versions list --model $MODEL_NAME

      #predictions on model
      # gcloud ml-engine predict --model $MODEL_NAME \
      #                          --version $VERSION_NAME \
      #                          --json-instances data/test/test_json_list.json \
      #                          > preds/test_json_list.txt

      #Create help facility
