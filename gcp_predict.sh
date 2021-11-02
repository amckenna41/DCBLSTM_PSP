################################################################################
###############            GCP Prediction Script                ################
################################################################################

#!/bin/bash

# add local predict functionality

#Help Funtion showing script usage
Help()
{
   echo ""
   echo ""
   echo "Basic Usage, using default parameters: ./gcp_training "
   echo "Usage: ./gcp_predict [--m|--i|--r|--v]"
   echo ""
   echo "Options:"
   echo "-m model used for prediction"
   echo "-i input data to be predicted using model"
   echo "-r cloud region to do prediction on"
   echo ""
   echo "-h     help"
   exit
}

#### Parse positonal arguments ###
#https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
#source ./ arguments.sh
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--model)
    MODEL="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--input_data)
    INPUT_DATA="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--region)
    REGION="$2"
    shift # past argument
    shift # past value
    ;;
    -v|--version)
    VERSION="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ $# -eq 0 ]
  then
    MODEL=
    INPUT_DATA=     #--json-instances, --json-request, --text-instances
    REGION="us-central1"
    VERSION=3.7
fi


if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 $1
fi

if [ -z "$MODEL" ]; then
  MODEL="default_model"
fi
if [ -z "$INPUT_DATA" ]; then
  INPUT_DATA="default_input"
fi
if [ -z "$REGION" ]; then
  REGION="default_region"
fi
if [ -z "$VERSION" ]; then
  VERSION="default_version"
fi

echo "Predicting with model on Google Cloud Platform"
echo "###################################################"
echo ""
echo "Job Details..."
echo "Model Name: $MODEL"
echo "Model Version: $VERSION"
echo "Region: $REGION"
echo "Input data: $INPUT_DATA"
echo ""
echo "###################################################"


#submitting Tensorflow prediction job to Google Cloud
gcloud ai-platform predict \
  --model=$MODEL \
  --version=$VERSION \
  --json-request=$INPUT_DATA \
  --region=$REGION
