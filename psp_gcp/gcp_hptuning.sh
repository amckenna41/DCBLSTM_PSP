
################################################################################
##########            GCP Hyperparameter Tuning Function             ###########
################################################################################

#!/bin/bash

#Job Parameters
BUCKET_NAME="keras-python-models-2"
PROJECT_NAME="ninth-optics-286313"
JOB_NAME="$RECURRENT_LAYER""_hp_config_model_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR="gs://keras-python-models-2"
LOGS_DIR="$JOB_DIR""/logs/tensorboard/hp_tuning_$(date +"%Y%m%d_%H%M")"
PACKAGE_PATH="training/"
STAGING_BUCKET="gs://keras-python-models-2"
HP_CONFIG="training/training_utils/hptuning_config.yaml"
MODULE="training.psp_rnn_gcp_hpconfig"
RUNTIME_VERSION="2.1"
PYTHON_VERSION="3.7"
REGION="us-central1"

#Checking pip update and update to mitigate warning messages
if !(pip --version == '20.2.1')
then
  echo "Updating pip to latest version"
  sudo -H pip3 install --upgrade pip
else
  echo "Pip up-to-date"
fi

#Function to parse hyperparameter config file
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml training/training_utils/hptuning_config.yaml)


echo "Hyperparameter tuning for LSTM model"
echo "Job Details..."
echo "Job Name: $JOB_NAME"
echo "Logs and model will be stored to $BUCKET_NAME bucket"
echo "Region: $REGION"
echo "Module: $MODULE"
echo "Runtime Version: $RUNTIME_VERSION"
echo ""
echo "GCP Machine Type Parameters..."
echo "Scale Tier: $trainingInput_scaleTier"
echo "Master Type: $trainingInput_masterType"
echo "Max Trials: $trainingInput_hyperparameters_maxTrials"
echo "Max Parellel Trials: $trainingInput_hyperparameters_maxParallelTrials"
echo "Enable Early Stopping: $trainingInput_hyperparameters_enableEarlyStopping"
echo ""

    #submit packaged training job to Gcloud Ai-Platform
    gcloud ai-platform jobs submit training $JOB_NAME \
      --package-path $PACKAGE_PATH \
      --module-name $MODULE \
      --staging-bucket $STAGING_BUCKET \
      --runtime-version $RUNTIME_VERSION \
      --python-version $PYTHON_VERSION  \
      --job-dir $JOB_DIR \
      --region $REGION \
      --config $HP_CONFIG \
      -- \
      --config_ $CONFIG


echo "To view model progress through tensorboard in Google Cloud shell or terminal execute..."
echo "tensorboard --logdir=$LOGS_DIR --port=8080"
echo "If in cloud shell, then click on the web preview option "

#visualise model results on TensorBoard
# tensorboard --logdir "gs://BUCKET_NAME/logs/tensorboard"

#Stream logs on command line
#gcloud ai-platform jobs stream-logs $JOB_NAME
