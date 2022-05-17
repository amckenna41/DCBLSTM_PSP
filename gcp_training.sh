
################################################################################
###############             GCP Training Script                 ################
################################################################################

#!/bin/bash

### check current version of pip and update, if neccessry ###
python3 -m pip install --user --upgrade pip

#automatically install any gcloud SDK updates
#yes | gcloud components update

#Help Funtion showing script usage
Help()
{
   echo "Bash Script for building and training PSP model's on GCP"
   echo ""
   echo "Basic Usage, using default parameters: ./gcp_training "
   echo "Usage: ./gcp_training [--config]"
   echo ""
   echo "Options:"
   echo "-h     help"
   echo "-config     Path to configuration file with all model and GCP training parameters"
   echo "-local      Flag that determines whether to train model on GCP or locally (0,1)"
   exit
}

for i in "$@"
do
case $i in
    -c=*|--config=*)
    CONFIG="${i#*=}"
    shift # past argument=value
    ;;
    -l=*|--local=*)
    LOCAL="${i#*=}"
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

#set default config file to dummy.json
if [ -z "$CONFIG" ]; then
  CONFIG="dummy.json"

fi

#set default local flag file to 0 (train on GCP)
if [ -z "$LOCAL" ]; then
  LOCAL=0

fi

#get path to configuration file depending on system OS
#append config filepath to user input config file
#remove any existing file extensions from user input config arg and append .json to it
ConfigPath() {

  # echo $pth | sed 's#/opt#c:#g'|sed 's#/#\\\\#g'
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CONFIG="psp_gcp/config/$(echo "$CONFIG" | cut -f 1 -d '.').json"
          # ...
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    CONFIG="psp_gcp/config/$(echo "$CONFIG" | cut -f 1 -d '.').json"

          # Mac OSX
  elif [[ "$OSTYPE" == "cygwin" ]]; then
    CONFIG="psp_gcp\\config\\$(echo "$CONFIG" | cut -f 1 -d '.').json"

  #         # POSIX compatibility layer and Linux environment emulation for Windows
  elif [[ "$OSTYPE" == "msys" ]]; then
    CONFIG="psp_gcp\\config\\$(echo "$CONFIG" | cut -f 1 -d '.').json"

  #         # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
  elif [[ "$OSTYPE" == "win32" ]]; then
    CONFIG="psp_gcp\\config\\$(echo "$CONFIG" | cut -f 1 -d '.').json"

  #         # I'm not sure this can happen.
  elif [[ "$OSTYPE" == "freebsd"* ]]; then
    CONFIG="psp_gcp\\config\\$(echo "$CONFIG" | cut -f 1 -d '.').json"

  else
        echo "Unknown OS, using default settings"
       CONFIG="psp_gcp\config\$(echo "$CONFIG" | cut -f 1 -d '.').json"
  fi
}

ConfigPath

#import environment variables from secrets script that stores sensitive GCP values
source psp_gcp/config/secrets.sh
#############################################################
#####                 secrets.sh                         ####
#!/usr/bin/env bash

# export PROJECT_ID=""
# export BUCKET=""
# export JOB_DIR=""
# export REGION=""
#############################################################

#extract sensitive env var values from secrets and set relevant keys in configuration json file using temp file
tmp=$(mktemp)
jq -r '.gcp_parameters[0].project_id = env.PROJECT_ID | .gcp_parameters[0].region = env.REGION |
  .gcp_parameters[0].bucket = env.BUCKET | .gcp_parameters[0].job_dir = env.JOB_DIR '  \
  $CONFIG > "$tmp" && mv "$tmp" $CONFIG

#parse GCP params from config json
GCP_PARAMS=$(jq -r .gcp_parameters[0] $CONFIG)
#parse params from config json
PARAMS=$(jq -r .parameters[0] $CONFIG)
#parse model params from config json
MODEL_PARAMS=$(jq -r .model_parameters[0] $CONFIG)

#parse all model and training parameters from JSON config file
CONFIG_FILENAME="$(basename -- $CONFIG)"
JOB_NAME="${CONFIG_FILENAME%.*}_""$(date +"%Y%m%d_%H%M")"

PROJECT_ID=$(jq -r .gcp_parameters[0].project_id $CONFIG)
PACKAGE_PATH=$(jq -r .gcp_parameters[0].package_path $CONFIG)
MODEL=$(jq -r .parameters[0].model $CONFIG)
MODULE=$(jq -r .gcp_parameters[0].module_name $CONFIG)
TRAINING_DATA=$(jq -r .parameters[0].training_data $CONFIG)
FILTERED=$(jq -r .parameters[0].filtered $CONFIG)
BATCH_SIZE=$(jq -r .parameters[0].batch_size $CONFIG)
EPOCHS=$(jq -r .parameters[0].epochs $CONFIG)
OPTIMIZER=$(jq -r .model_parameters[0].optimizer.name $CONFIG)
LEARNING_RATE=$(jq -r .model_parameters[0].optimizer.learning_rate $CONFIG)
LOGS_PATH=$(jq -r .parameters[0].logs_path $CONFIG)
TEST_DATASET=$(jq -r .parameters[0].test_dataset $CONFIG)
BUCKET=$(jq -r .gcp_parameters[0].bucket $CONFIG)
RUNTIME_VERSION=$(jq -r .gcp_parameters[0].runtime_version $CONFIG)
PYTHON_VERSION=$(jq -r .gcp_parameters[0].python_verion $CONFIG)
JOB_DIR=$(jq -r .gcp_parameters[0].job_dir $CONFIG)
REGION=$(jq -r .gcp_parameters[0].region $CONFIG)
SCALE_TIER=$(jq -r .gcp_parameters[0].scale_tier $CONFIG)
MACHINE_TYPE=$(jq -r .gcp_parameters[0].master_machine_type $CONFIG)
CUDA=$(jq -r .gcp_parameters[0].cuda $CONFIG)
TPU=$(jq -r .gcp_parameters[0].tpu $CONFIG)
TENSORBOARD=$(jq -r .model_parameters[0].callbacks.tensorboard.tensorboard $CONFIG)
EARLY_STOPPING=$(jq -r .model_parameters[0].callbacks.earlyStopping.earlyStopping $CONFIG)
MODEL_CHECKPOINT=$(jq -r .model_parameters[0].callbacks.modelCheckpoint.modelCheckpoint $CONFIG)
LR_SCHEDULER=$(jq -r .model_parameters[0].callbacks.lrScheduler.lrScheduler $CONFIG)
LEARNING_RATE_SCHEDULER=$(jq -r .model_parameters[0].callbacks.lrScheduler.scheduler $CONFIG)
CSV_LOGGER=$(jq -r .model_parameters[0].callbacks.csv_logger.csv_logger $CONFIG)
REDUCE_LR_ON_PLATEAU=$(jq -r .model_parameters[0].callbacks.reduceLROnPlateau.reduceLROnPlateau $CONFIG)

if [ $CUDA -eq 1 ]; then
  # CONFIG="dummy.json"
  CUDA=0
  CUDA_VISIBLE_DEVICES=0
else
  CUDA=1
  CUDA_VISIBLE_DEVICES=1
fi
export CUDA_VISIBLE_DEVICES=0   # - initialise CUDA env var
# export CUDA_VISIBLE_DEVICES=1 - If using 1 CUDA enabled GPU

#if using TPU or not
if [ $TPU -eq 1 ]; then
  echo ""
  if [ "$SCALE_TIER" != "BASIC_TPU" ]; then
    TPU=0
  fi
fi

if [ $LOCAL -eq 0 ]; then
  echo "Running model on Google Cloud Platform..."
else
  echo "Running $MODEL locally..."
fi

echo "###################################################"
echo "Job Details:"
echo ""
echo "Configuration File: $CONFIG"
echo "Project ID: $PROJECT_ID"
echo "Package Name: $PACKAGE_PATH"
echo "Entry Module: $MODULE"
echo "Job Name: $JOB_NAME"
echo "Staging Bucket: $BUCKET"
echo "Cloud Runtime Version: $RUNTIME_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "Region: $REGION"
echo "Scale Tier: $SCALE_TIER"
echo "Machine Type: $MACHINE_TYPE"
echo "Logs and models stored in bucket: $JOB_DIR"
echo ""
echo "###################################################"
echo "Training Details:"
echo ""
echo "Model name: $MODEL"
echo "CullPDB Training Data: $TRAINING_DATA"
echo "Using filtered training data: $FILTERED"
echo "Test Dataset: $TEST_DATASET"
echo "Number of epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Optimizer: $OPTIMIZER"
echo "Learning Rate: $LEARNING_RATE"
echo "Using Cuda: $CUDA_VISIBLE_DEVICES"
echo "Callbacks:"
echo "  Tensorboard: $TENSORBOARD"
echo "  Early Stopping: $EARLY_STOPPING"
echo "  Model Checkpoint: $MODEL_CHECKPOINT"
echo "  Learning Rate Scheduler: $LR_SCHEDULER"
echo "  Scheduler: $LEARNING_RATE_SCHEDULER"
echo "  CSV Logger: $CSV_LOGGER"
echo "  Reduce LR on plateau: $REDUCE_LR_ON_PLATEAU"
echo ""
echo "###################################################"

#get current project set on gcloud sdk
CURRENT_PROJECT=$(gcloud config list --format 'value(core.project)')

#if current project configured != user input Project id then set project config
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
  gcloud config set project $PROJECT_ID
fi
if [[ $LOCAL -eq 0 ]]
  then
  #submitting Tensorflow training job to Google Cloud
   gcloud ai-platform jobs submit training $JOB_NAME \
       --package-path $PACKAGE_PATH \
       --module-name $MODULE \
       --staging-bucket $BUCKET \
       --runtime-version $RUNTIME_VERSION \
       --python-version $PYTHON_VERSION  \
       --job-dir $JOB_DIR \
       --region $REGION \
       --scale-tier $SCALE_TIER \
       --master-machine-type $MACHINE_TYPE \
       -- \
       --local $LOCAL \
       --config $CONFIG \
       --gcp_params "$GCP_PARAMS" \
       --params "$PARAMS" \
       --model_params "$MODEL_PARAMS"

  #stream logs to terminal
  # gcloud ai-platform jobs stream-logs $JOB_NAME

elif [[ $LOCAL -eq 1 ]]
then

  gcloud ai-platform local train \
    --module-name $MODULE \
    --package-path $PACKAGE_PATH \
    --job-dir $JOB_DIR \
    -- \
    --local $LOCAL \
    --config $CONFIG \
    --gcp_params "$GCP_PARAMS" \
    --params "$PARAMS" \
    --model_params "$MODEL_PARAMS"

    #if error running model locally, ensure correct local python version is pointed to by ml-engine
    #gcloud config set ml_engine/local_python $(which python3)
fi

#reset sensitive GCP parameter values to null after training completed
jq -r '.gcp_parameters[0].project_id = "" | .gcp_parameters[0].region = "" |
  .gcp_parameters[0].bucket = "" | .gcp_parameters[0].job_dir = "" '  \
  $CONFIG > "$tmp" && mv "$tmp" $CONFIG


################################################################################

# echo "To view model progress through tensorboard in Google Cloud shell or terminal execute..."
# echo "tensorboard --logdir=$LOGS_DIR --port=8080"
# echo "If in cloud shell, then click on the web preview option "


# ##############################################
#
###       Results Notification Function    ###
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
#    -e OBJECT_FINALIZE $BUCKET




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

      # Explicitly tell `gcloud ai-platform local train` to use Python 3
      # ! gcloud config set ml_engine/local_python $(which python3)

      # ! gcloud ai-platform models create $MODEL_NAME \
      # --regions $REGION

      # ! gcloud ai-platform versions create $MODEL_VERSION \

      # make default region so user doesn't have to put in numeric choice
      # !gcloud config set ai_platform/region us-central1

      #Uploading objects to GCP Storage
      #gsutil cp OBJECT_LOCATION gs://DESTINATION_BUCKET_NAME/
