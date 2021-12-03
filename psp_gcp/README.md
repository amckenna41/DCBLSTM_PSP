# Google Cloud Platform Distribution for Protein Structure Prediction <a name="TOP"></a>

A full GCP pipeline for building, training and evaluating any of the models used in this project.

Table of Contents
-----------------

* [Setup](#Setup)
* [Implementation](#implementation)
* [Requirements](#requirements)
* [Running Locally](#)
* [Running on GCP](#)

![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

Setup
--------

### Set up local development environment ###


The Google Cloud guide to [Setting up a Python development environment](https://cloud.google.com/python/setup) provide detailed instructions for meeting these requirements. The following steps provide a condensed set of instructions:

 1. Install and initialize the Cloud SDK.

 2. Install Python 3.

 3. Install virtualenv and create a virtual environment that uses Python 3. virtualenv mypython

 4. Activate that environment. source mypython/bin/activate


### Set up your GCP project ###

 1. Select or create a GCP project.

 2. Make sure that billing is enabled for your project.

 3. Enable the AI Platform ("Cloud Machine Learning Engine") and Compute Engine APIs.

 4. From a command line/terminal run:
```
gcloud config set project $PROJECT_ID
```
where $PROJECT_ID is your GCP project ID.

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

1. Run the following code to create bucket using gsutil tool:
```
gsutil mb -l $REGION gs://$BUCKET_NAME
```
where $REGION is GCP region and $BUCKET_NAME is the name of the bucket.

## Usage ##

Using a terminal/command line, ensure that the current working directory is the main DCBLSTM_PSP.  <br>

To call the model with the optimum parameters, from a command line, run:

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

To get secondary structure prediction of protein sequence from pre-built model, from a command line, run:
```
./gcp_predict

-m model
-i input_data

e.g. ./gcp_predict -m psp_dculstm_gcp_model -i input_predictions.json
```

## AI-Platform Configuration ## **update this

The configuration  ... can be found in gcp_training_config.yaml . For running the main CDBLSTM/CDULSTM models the high memory CPU n1-highmem-8 machine is sufficent. (run on compute engine)
GPU's and TPU's are also available which were tested with the models but ultimately gave similar results to using just CPU's but at a greater cost, therefore high memory CPU machines were utilised.

More info about the different GCP machine types can be found at:
https://cloud.google.com/compute/docs/machine-types



## Google Cloud Platform Architecture ##
The cloud architecture used within the GCP for this project can be seen below were several services were taken advantage of including: Ai-Platform, Compute Engine GCS, Logging, Monitoring and IAM.

![alt text](https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/gcp_architecture.png?raw=true)

## Model Saving Structure ##


* After training is complete, the model and all its attributes and associated objects will be stored in a new folder inside a GCP storage bucket. The directory structure of this can be seen below.
* The folder name in the bucket is determined by the Ai-Platform job name, which itself is created once the training script is called in the format: $MODEL_NAME_YYYY_MM_DD:HH:MM_epochs_$EPOCHS_batch_size_$BATCH_SIZE

```
bucket_name
├── job_name
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

Requirements
-------------

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

Using GCP functionality for Workflow
------------------------------------
1. Encrypting Service Account json using gpg tool on command-line.
gpg --symmetric --cipher-algo AES256 service-account.json

Files and Directory structure
-----------------------------

* `/config` -
* `/models` -
* `/notification_func` -
* `/training` -
* 'gcp_hptuning.sh' -
* 'gcp_notification_func.sh' -
* 'gcp_predict.sh' -
* 'gcp_training.sh' -
* 'MANIFEST.in' -
* 'setup.py'


https://cloud.google.com/compute/docs/machine-types




[Back to top](#TOP)

#Set the ml-engine to point to the correc pythong3 distro
#gcloud config set ml_engine/local_python $(which python3)
