#!/bin/bash

if !(pip --version == '20.2.1')
then
  echo "Updating pip to latest version"
  sudo -H pip3 install --upgrade pip
else
  echo "Pip up-to-date"
fi

JOB_NAME="lstm_model_$(date +"%Y%m%d_%H%M%S")"
BUCKET_NAME="keras-python-models"
JOB_DIR="gs://keras-python-models"
PACKAGE_PATH="training/"
STAGING_BUCKET="gs://keras-python-models"
CONFIG="training/gcp_training_gpu.yaml"
MODULE="training.psp_lstm_gcp_model"
RUNTIME_VERSION="2.1"
PYTHON_VERSION="3.7"
REGION="us-central1"

#Function to parse config file
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
eval $(parse_yaml training/gcp_training_gpu.yaml)

# gcloud ai-platform jobs submit training JOB75 --package-path training/ --module-name training.task --staging-bucket gs://keras-python-models --region us-central1 --config training/temp_gcp_configfile.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
# gcloud ai-platform jobs submit training JOB99 --package-path training/ --module-name training.task --staging-bucket gs://keras-python-models --region us-central1 --config training/cloudml-gpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
echo "Running LSTM model on Google Cloud..."
echo "Job Details..."
echo "Job Name: $JOB_NAME"
echo "Cloud Runtime Version: $RUNTIME_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "Region: $REGION"
echo "Logs and models stored in bucket: $JOB_DIR"
echo ""

echo "GCP Machine Type Parameters..."
echo "Parsing YAML"

eval $(parse_yaml training/gcp_training_gpu.yaml)

echo "Scale Tier: $trainingInput_scaleTier"
echo "Master Type: $trainingInput_masterType"
echo "Worker Type: $trainingInput_workerType"
echo "Parameter Server Type: $trainingInput_parameterServerType"
echo "Worker Count : $trainingInput_workerCount"
echo "Parameter Server Count: $trainingInput_parameterServerCount"

gcloud ai-platform jobs submit training $JOB_NAME \
 --package-path $PACKAGE_PATH \
 --module-name $MODULE \
 --staging-bucket $STAGING_BUCKET \
 --region $REGION \
 --config $CONFIG \
 --runtime-version $RUNTIME_VERSION \
 --python-version $PYTHON_VERSION \
 --job-dir $JOB_DIR
