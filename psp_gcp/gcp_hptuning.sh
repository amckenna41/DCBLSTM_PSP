#!/bin/bash

#Script called on hyperparameter tuning model class
## Optimal Default values for model hyperparameters ##
#   Edit script with desired hyperparameters

#Hyperparameters for CNN component of model
CONV1_FILTERS=16
CONV2_FILTERS=32
CONV3_FILTERS=64
WINDOW_SIZE=7
CONV2D_DROPOUT=0.4
KERNEL_REGULARIZER='l2'
CONV_KERNEL_INITIALIZER='glorot_uniform'
POOL_SIZE=2

#Hyperparameters for RNN component of model
RECURRENT_LAYER1=300
RECURRENT_LAYER2=300
RECURRENT_DROPOUT=0.5
RECURRENT_RECURRENT_DROPOUT=0.4
AFTER_RECURRENT_DROPOUT=0.4
BIDIRECTION=True
RECURRENT_LAYER='lstm'
RECURRENT_KERNEL_INITIALIZER='glorot_uniform'

#Hyperparameters for DNN component of model
DENSE_1=600
DENSE_DROPOUT=0.5
DENSE_KERNEL_INITIALIZER='glorot_uniform'

#Model hyperparameters
OPTIMIZER="adam"
LEARNING_RATE=0.003
EPOCHS=3
BATCH_SIZE=120
ALL_DATA=1.0

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
            --conv1_filters $CONV1_FILTERS \
            --conv2_filters $CONV2_FILTERS \
            --conv3_filters $CONV3_FILTERS \
            --window_size $WINDOW_SIZE \
            --kernel_regularizer $KERNEL_REGULARIZER \
            --conv_weight_initializer $CONV_KERNEL_INITIALIZER \
            --pool_size $POOL_SIZE \
            --conv2d_dropout $CONV2D_DROPOUT \
            --recurrent_layer1 $RECURRENT_LAYER1 \
            --recurrent_layer2 $RECURRENT_LAYER2 \
            --recurrent_dropout $RECURRENT_DROPOUT \
            --after_recurrent_dropout $AFTER_RECURRENT_DROPOUT \
            --recurrent_recurrent_dropout $RECURRENT_RECURRENT_DROPOUT \
            --recurrent_weight_initializer $RECURRENT_KERNEL_INITIALIZER \
            --optimizer $OPTIMIZER \
            --learning_rate $LEARNING_RATE \
            --bidirection $BIDIRECTION \
            --recurrent_layer $RECURRENT_LAYER \
            --dense_1 $DENSE_1 \
            --dense_weight_initializer $DENSE_KERNEL_INITIALIZER \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --alldata $ALL_DATA \
            --logs_dir $LOGS_DIR \
            --project_name $PROJECT_NAME \
            --job_name $JOB_NAME


echo "To view model progress through tensorboard in Google Cloud shell or terminal execute..."
echo "tensorboard --logdir=$LOGS_DIR --port=8080"
echo "If in cloud shell, then click on the web preview option "
#visualise model results on TensorBoard
# tensorboard --logdir "gs://BUCKET_NAME/logs/tensorboard"

#Stream logs on command line
#gcloud ai-platform jobs stream-logs $JOB_NAME



#finish help function
# helpFunction()
# # https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
# {
#    echo ""
#    echo "Usage: $0 -epochs epochs -batch_size batch size -alldata alldata"
#    echo -e "\t-epochs Number of epochs to run with model"
#    echo -e "\t-batch_size Batch Size to use with model"
#    echo -e "\t-alldata What proportion of data to use"
#    # exit 1 # Exit script after printing help
# }
# helpFunction

# while getopts "epochs:batch_size:alldata:" opt
# do
#    case "$opt" in
#       epochs ) epochs="$OPTARG" ;;
#       batch_size ) batch_size="$OPTARG" ;;
#       alldata ) alldata="$OPTARG" ;;
#       ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
#    esac
# done
#
# # Print helpFunction in case parameters are empty
# if [ -z "$epochs" ] || [ -z "$batch_size" ] || [ -z "$alldata" ]
# then
#    echo ""
#    echo "Some or all of the parameters are empty";
#    echo "Setting parameters to default values"
#    helpFunction
# fi
