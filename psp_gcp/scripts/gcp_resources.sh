#Shell script for making all the resources required for deployment
PROJECT_ID='[PROJECT_ID]' #@param {type:"string"}
REGION='[REGION_NAME]' #e.g 'us-central1', 'europe-west1'
gcloud auth login

gcloud init

gcloud version

gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID


gsutil mb -l $REGION gs://$BUCKET_NAME
#list bucket contents
#get size of bucket
#create folders in bucket
#enable API Access

#upload data to google bucket 

#create checkpoints directory in bucket
#create models directory in bucket
