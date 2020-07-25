gcloud ml-engine jobs submit training JOB26 --package-path ./training --module-name training.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config training/cloudml-gpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
gcloud ml-engine jobs submit training TPU_JOB1 --package-path ./trainer --module-name trainer.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config trainer/cloudml-tpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models


# gcloud ai-platform jobs submit training JOB26 --package-path ./training --module-name training.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config training/cloudml-gpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
# gcloud ai-platform jobs submit training TPU_JOB1 --package-path ./trainer --module-name trainer.psp_lstm_gcp --staging-bucket gs://keras-python-models --region us-central1 --config trainer/cloudml-tpu.yaml --runtime-version 2.1 --python-version 3.7  --job-dir gs://keras-python-models
