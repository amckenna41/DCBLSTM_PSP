#!/bin/bash

#check current version of pip and update if neccessry
if !(pip --version == '20.2.2')
then
  echo "Updating pip to latest version"
  sudo -H pip3 install --upgrade pip
else
  echo "Pip up-to-date"
fi

#update Python Path
# export PATH="${PATH}:/root/.local/bin"
export PYTHONPATH="${PYTHONPATH}:/root/.local/bin"

# export GOOGLE_APPLICATION_CREDENTIALS="service-account.json" - set GAC env variable
# echo $GOOGLE_APPLICATION_CREDENTIALS

#set arguments to be passed into model
BATCH_SIZE=120
EPOCHS=5
ALL_DATA=1.0

#set Ai-Platform Job environment variables
BUCKET_NAME="gs://keras-python-models-2"
JOB_NAME="CDBLSTM_+model$(date +"%Y%m%d_%H%M")_epochs_""$EPOCHS""_batch_size_""$BATCH_SIZE"
JOB_DIR="$BUCKET_NAME/job_logs"      # - where to store job logs
PACKAGE_PATH="training/"             # - path of folder to be packaged
CONFIG="training/training_utils/gcp_training_config.yaml"   # - job config file
MODULE="training.psp_blstm_gcp_model"       # - main calling module
RUNTIME_VERSION="2.1"   # - https://cloud.google.com/ai-platform/training/docs/runtime-version-list
PYTHON_VERSION="3.7"
REGION="us-central1"    # - cloud region to run job
CUDA_VISIBLE_DEVICES=""   # - initialise CUDA env var
LOGS_DIR="$JOB_DIR""/logs/tensorboard/$JOB_NAME"   # - TensorBoard logs

#Function to parse GCP config file
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
eval $(parse_yaml training/training_utils/gcp_training_config.yaml)

echo "Running LSTM model on Google Cloud..."
echo "Job Details..."
echo "Job Name: $JOB_NAME"
echo "Cloud Runtime Version: $RUNTIME_VERSION"
echo "Python Version: $PYTHON_VERSION"
echo "Region: $REGION"
echo "Logs and models stored in bucket: $JOB_DIR"
echo ""
echo "GCP Machine Type Parameters..."

echo "Scale Tier: $trainingInput_scaleTier"
echo "Master Type: $trainingInput_masterType"
echo "Worker Type: $trainingInput_workerType"
echo "Parameter Server Type: $trainingInput_parameterServerType"
echo "Worker Count : $trainingInput_workerCount"
echo "Parameter Server Count: $trainingInput_parameterServerCount"
echo ""

     #submitting keras training job to Google Cloud
     gcloud ai-platform jobs submit training $JOB_NAME \
         --package-path $PACKAGE_PATH \
         --module-name $MODULE \
         --staging-bucket $BUCKET_NAME \
         --runtime-version $RUNTIME_VERSION \
         --python-version $PYTHON_VERSION  \
         --job-dir $JOB_DIR \
         --region $REGION \
         --config $CONFIG \
         -- \
         --epochs $EPOCHS \
         --batch_size $BATCH_SIZE \
         --alldata $ALL_DATA \
         --logs_dir $LOGS_DIR
         # --job_name $JOB_NAME   ****

echo "To view model progress through tensorboard in Google Cloud shell or terminal execute..."
echo "tensorboard --logdir=$LOGS_DIR --port=8080"
echo "If in cloud shell, then click on the web preview option "

###Visualise model results on TensorBoard###
# tensorboard --logdir [LOGS_PATH] - path declared in Tensorboard callback:
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)



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
      #   --alldata $ALL_DATA \
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
