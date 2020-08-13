#Script called on hyperparameter tuning model class
#Edit script with desired hyperparameters

CONV2D_LAYER1_FILTERS=42
CONV2D_LAYER2_FILTERS=84
CONV2D_ACTIVATION="relu"
CONV2D_DROPOUT=0.5
LSTM_LAYER1_NODES=400
LSTM_LAYER2_NODES=300
LSTM_DROPOUT=0.5
LSTM_RECURRENT_DROPOUT=0.5
AFTER_LSTM_DROPOUT=0.5
OPTIMIZER="adam"
LEARNING_RATE=0.003
EPOCHS=10
BATCH_SIZE=42

BUCKET_NAME="keras-python-models"
JOB_NAME="lstm_hp_config_model_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR="gs://keras-python-models"
PACKAGE_PATH="training/"
STAGING_BUCKET="gs://keras-python-models"
HP_CONFIG="training/hptuning_config.yaml"
MODULE="training.psp_lstm_gcp_hpconfig"
# RUNTIME_VERSION="2.1"
RUNTIME_VERSION="1.15"
PYTHON_VERSION="3.7"
REGION="us-central1"
#!/bin/bash

echo "Hyperparameter tuning for LSTM model"
echo "Job Details..."
echo "Job Name: $JOB_NAME"
echo "Logs and model will be stored to $BUCKET_NAME bucket"
echo "Region: $REGION"
echo ""
echo "Hyperparameters..."
echo ""
echo "GCP Machine Type Parameters..."

pip --version
sudo -H pip3 install --upgrade pip

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
eval $(parse_yaml training/hptuning_config.yaml)


# SCALE_TIER=STANDARD_1
# CONV2D_LAYER1_FILTERS =
# export CONV2D_LAYER1_FILTERS=42
# export CONV2D_LAYER2_FILTERS=84
# export CONV2D_ACTIVATION='relu'
# (conv2d_layer1_filters=42, conv2d_layer2_filters=84,
#                             conv2d_activation='relu', conv2d_dropout=0.5, lstm_layer1_nodes=400, lstm_layer2_nodes=300,
#                             lstm_dropout=0.5, lstm_recurrent_dropout=0.5, after_lstm_dropout=0.4, optimizer='adam'):
# echo $CONV2D_LAYER1_FILTERS
gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path training/ \
  --module-name training.psp_lstm_gcp_hpconfig \
  --staging-bucket gs://keras-python-models \
  --region us-central1 \
  --config training/hptuning_config.yaml \
  --runtime-version 1.15 \
  --python-version 3.7  \
  --job-dir gs://keras-python-models
  # -- \          #user_args from here
  # --conv2d_dropout 0.5
  # --

  #

#Stream logs on command line
#gcloud ai-platform jobs stream-logs $JOB_NAME
