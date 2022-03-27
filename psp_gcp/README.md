# Google Cloud Platform Distribution for Protein Structure Prediction <a name="TOP"></a>

A full GCP pipeline for building, training and evaluating any of the models used in this project.

![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

Table of Contents
-----------------

* [Setup](#Setup)
* [Implementation](#implementation)
* [Requirements](#requirements)
* [Running Locally](#)
* [Running on GCP](#)

Setup
--------

### Set up local development environment ###

<details>
The Google Cloud guide to [Setting up a Python development environment](https://cloud.google.com/python/setup) provide detailed instructions for meeting these requirements. The following steps provide a condensed set of instructions:

 1. Install and initialize the Cloud SDK.

 2. Install Python 3.

 3. Install virtualenv and create a virtual environment that uses Python 3:
 python3 -m venv psp_venv

 4. Activate that environment: source psp_venv/bin/activate
 </details>


### Set up your GCP project ###

<details>
 1. Select or create a GCP project.

 2. Make sure that billing is enabled for your project.

 3. Enable the AI Platform ("Cloud Machine Learning Engine") and Compute Engine APIs.

 4. From a command line/terminal run:
```
gcloud config set project $PROJECT_ID
```
where $PROJECT_ID is your GCP project ID.

</details>


### Authenticate GCP account ###

<details>
<summary>Click to see how to authenticate on GCP</summary>

1. In the GCP Console, go to the Create service account key page.

1. From the Service account drop-down list, select New service account.

2. In the Service account name field, enter a name.

3. From the Role drop-down list, select Machine Learning Engine > AI Platform Admin and Storage > Storage Object Admin.

4. Click Create. A JSON file that contains your key downloads to your local environment.

5. Enter the path to your service account key as the GOOGLE_APPLICATION_CREDENTIALS environment variables as below:
```
export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
```
</details>

### Create Bucket ###

<details>

1. Run the following code to create bucket using gsutil tool:
```
gsutil mb -l $REGION gs://$BUCKET_NAME
```
where $REGION is GCP region and $BUCKET_NAME is the name of the bucket.
</details>

### Enable GCP API's ###

You will need to enable several API's to use and communicate with GCP's services including for storage, authentication, running the models and for logging/monitoring purposes. 

1. To list all of the GCP API's available, run the following command:
```
gcloud services list
```

2. To enable each API service, run the following, replacing the API with its respective command:
```
gcloud services enable storage.googleapis.com
```
* Compute Engine API - compute.googleapis.com 
* IAM API - iam.googleapis.com
* Cloud Logging API - logging.googleapis.com
* AI Platform training & prediction API - ml.googleapis.com
* Cloud Monitoring API - monitoring.googleapis.com
* Storage API - storage.googleapis.com

Usage
-----

Using a terminal/command line, ensure that the current working directory is the main DCBLSTM_PSP.  <br>

**To call the model, from a command line, run gcp_training script:**

```
e.g ./gcp_training.sh --config=dcblstm.json --local=0

--config: relative path to desired model config file to train.
--local: whether to run the GCP pipeline locally or on GCP (0 - run on GCP, 1 - run locally).
```

<!-- To call the hyperparameter tuning script, from a command line call:
```
./gcp_hptuning
```
If you want to change any of the default hyperparameters then pass the parameter in when calling the script, e.g:
```
./gcp_hptuning
-b batch size (default = 120)
-e epochs (default = 5)
-td test dataset (default = cb513)

e.g.
./gcp_hptuning -e 10 -b 120 -td casp10
``` -->

**To get secondary structure prediction of protein sequence from pre-built model, from a command line, run:**
```
./gcp_predict --m="model.h5" --i="data.json"

-m model to use for prediction
-i input_data to predict secondary structure for
```

AI-Platform Configuration
-------------------------

The json configuration files contains parameters specifically for configuring the job on Google's AI-Platform service. For example, the GCP configuration parameters for the dcblstm.json is below:

```
"gcp_parameters": [
  {
    "project_id": "",
    "package_path": "psp",
    "module_name": "psp.main_gcp",
    "bucket": "",
    "runtime_version": "2.1",
    "python_verion": "3.7",
    "job_dir": "",
    "region": "",
    "scale_tier": "CUSTOM",
    "master_machine_type": "n1-highmem-8",
    "cuda": 0
  }
]
```

As you can see, the GCP Project ID, bucket name, job directory and region are initally empty. This was a personal design and security choice as I did not want to upload these to the repo therefore they are imported from a secrets.sh script. This script is stored in the config directory and imported in the main gcp_training script using:

```
source psp_gcp/config/secrets.sh
```
The secrets.sh script is in the format:

```
export PROJECT_ID=""
export BUCKET=""
export JOB_DIR=""
export REGION=""
```

For the larger more computationally expensive models, such as any of the recurrent models, the n1-highmem-8 machine was used which consists of 8 vCPU's each with 52GB of memory. Furthermore, for the smaller less memory intensive models, the n1-highmem-4 or n1-highmem-2 models machines were used. More info about the different GCP machine types can be found at: <br>
https://cloud.google.com/compute/docs/machine-types


Google Cloud Platform Architecture
----------------------------------

The cloud architecture used within the GCP for this project can be seen below where several services were taken advantage of including: Ai-Platform, Compute Engine GCS, Logging, Monitoring and IAM.

<p align="center">
<img src="images/gcp_architecture.png" alt="arch"/>
</p>


<!-- ![alt text](https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/gcp_architecture.png?raw=true) -->

Model Saving Structure
----------------------

The model saving structure follows that of the locally trained models:

```
bucket_name
├── job_name_DDMMYYY:MM
|   └── model_logs
|   └── model_checkpoints
│   └── model_plots         
│         └── figure1.png
│         └── figure2.png
|         └── ....png
│   └── model.h5
|   └── model.png
│   └── model_history.pckl
│   └── model_arch.json
│   └── model_output.csv
|   └── model_config.json
|   └── training.log
└-
```
<!-- * After training is complete, the model and all its attributes and associated objects will be stored in a new folder inside a GCP storage bucket. The directory structure of this can be seen below.
* The folder name in the bucket is determined by the Ai-Platform job name, which itself is created once the training script is called in the format: $MODEL_NAME_YYYY_MM_DD:HH:MM_epochs_$EPOCHS_batch_size_$BATCH_SIZE -->


GCP Notification Function
-------------------------

This Google Cloud Function is used to notify when training has been completed and notifies some of the training results via an email server to a recipient. Currently, with Google Cloud's ML Engine/Ai-Platform there is no mechanism at which to know when training has been completed and the Ai job is finished, this function alleviates that.

**To configure function, call setup script:**
```
./gcp_notification_func $BUCKET_NAME $TOPIC $SUBSCRIPTION $SOURCE_DIR $TOMAIL $FROMMAIL $EMAIL_PASS

$BUCKETNAME - name of GCP storage bucket
$TOPIC - name of pubsub topic
$SUBSCRIPTION - name of pubsub subscription
$SOURCE_DIR - name of dir containing function source code (main.py & requirements.txt)
$TOMAIL - receipient of training notification and results
$FROMMAIL - sender of training notification, ideally a gmail account with less secure app access configured
$EMAIL_PASS - password for less secure $FROMMAIL email account
```

Using GCP functionality for Workflow
------------------------------------
1. Encrypting Service Account json using gpg tool on command-line.
gpg --symmetric --cipher-algo AES256 service-account.json

2. Input passphrase when prompted.

3. Upload secrets to repo environment variables

4. Import passphrase secret as environment variable in workflow as:
```
env:
  PASSPHRASE: ${{ secrets.PASSPHRASE }}
```
5. Execute decrypt_secret.sh script
```
./.github/scripts/decrypt_secret.sh
```

Directory structure
-------------------

* `/config` - configuration json files for GCP code pipeline.
* `/notification_func` - directory for notification func used to notify user when training has completed/failed.


[Back to top](#TOP)
