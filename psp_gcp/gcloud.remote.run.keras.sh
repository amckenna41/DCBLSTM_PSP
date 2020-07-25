

gcloud ai-platform jobs submit training JOB1
--package-path=training
--module-name=training.psp_lstm_gcp
#--staging-bucket="${TRAIN_BUCKET}"   \
--job-dir=gs://keras-python-models
--region=us-central1
--config=training/cloudml-gpu.yaml
#

# TRAINER_PACKAGE_PATH="path-to-your-application-sources"
# MAIN_TRAINER_MODULE="trainer-task"
# PACKAGE_STAGING_PATH="path-to-your-chosen-staging-bucket"
# JOB_NAME="your-job-name"
# JOB_DIR="path-to-your-job-output-bucket"
# REGION="region"
