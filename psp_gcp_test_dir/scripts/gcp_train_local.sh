#Script to run keras model locally

MODULE=
PACKAGE_PATH=
JOB_DIR=
STORAGE_BUCKET=

echo "Running Keras model locally..."
gcloud config set ml_engine/local_python $(which python3)
gcloud ai-platform local train \
  --module-name training.task \
  --package-path training/
  --job-dir
