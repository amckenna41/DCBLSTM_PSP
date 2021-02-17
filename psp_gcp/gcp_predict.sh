

gcloud components update

#### Parse positonal arguments ###
#https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
#source ./ arguments.sh
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--model)
    BATCH_SIZE="$2"
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
    -td|--test_dataset)
    TEST_DATASET="$2"
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


echo "Predicting with model on Google Cloud Platform"
echo ""
echo "Job Details..."
echo "Model Name: $MODEL"
echo "Model Version: $VERSION"
echo "Region: $REGION"
echo "Input data: $INPUT_DATA"
echo ""

#submitting Tensorflow prediction job to Google Cloud
gcloud ai-platform predict \
  --model=$MODEL \
  --version=$VERSION \
  --json-request=$INPUT_DATA \
  --region=$REGION
