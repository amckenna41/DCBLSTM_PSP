#!/bin/bash

#Script for configuring connection to the Google Cloud to be able to train and evaluate
#model in GCP

# export PROJECT_ID="project_id"
# export REGION_NAME="region_name"
export SERVICE_ACCOUNT_KEY_PATH="service_account_key_path"
export SERVICE_ACCOUNT_NAME="service_account_name"

#1: Install Google Cloud SDK
#install SDK in interactive mode
# curl https://sdk.cloud.google.com | bash

#install sdk in non-interactive mode, save install script locally
# curl https://sdk.cloud.google.com > install.sh
# bash install.sh --disable-prompts

#2. Authenticate with user account - You must create a GCP account & project prior to this step
#authenticate GCP SDK, credentials will be automatically stored in config file
#where the SDK looks for the relevant credentials when using sdk tools
# gcloud auth login
#gcloud auth application-default login - authenticate via browser with Google Account


#2a) Authenticate using a service account
# gcloud auth activate-service-account --key-file service-account.json [KEY_FILE_PATH]
# gcloud auth activate-service-account [ACCOUNT] --key-file=[KEY_FILE_PATH]
# # gcloud iam service-accounts keys create - create key for existing service account, save to cwd

#list accounts with credentials on the current system
gcloud auth list

#create configuration
#gcloud config configurations create [NAME]

#switch active account
#gcloud config set account [ACCOUNT] - [ACCOUNT - full email address of account]

#switch accounts by creating seperate configuration that specifies different account
gcloud config configurations activate [CONFIGURATION]

#revoke credentials for an account
gcloud auth revoke [ACCOUNT ]

#view relevant GCP and SDK info
gcloud info

#check network connectivity
gcloud info --run-diagnostics

#set project property in configuration
gcloud config set project PROJECT_ID

#set region in configuration
gcloud config set compute/zone REGION_NAME
#view properties of current active configuration
gcloud config list

#list all current configurations
gcloud config configurations list

#delete configuration
gcloud config configurations delete [NAME]

#get current version of sdk
gcloud version
#configuring SDK behind proxy/firewall - https://cloud.google.com/sdk/docs/proxy-settings


#3.) Initialise gcloud sdk
# gcloud init
# gcloud auth activate-service-account
#gcloud auth application-default login


# 4.) Enable API Access - Go to APIs and Services, Enable APIs and Services,
#enable AI Platform Training & Prediction API, Compute Engine API, Cloud Logging API


#Full step-by-step setup and configuration for the GCP and SDK can be found in the docs:
#https://cloud.google.com/sdk/docs/
#
# #!/bin/bash
#
# URL=https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
#
# function download {
#   scratch="$(mktemp -d -t tmp.XXXXXXXXXX)" || exit
#   script_file="$scratch/install_google_cloud_sdk.bash"
#
#   echo "Downloading Google Cloud SDK install script: $URL"
#   curl -# "$URL" > "$script_file" || exit
#   chmod 775 "$script_file"
#
#   echo "Running install script from: $script_file"
#   "$script_file" "$@"
# }
#
# download "$@"


##Windows GCP Configuration ##
# https://cloud.google.com/sdk/docs/install#windows

#From windows Powershell, to download and launch SDK installation:

# (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
#
# & $env:Temp\GoogleCloudSDKInstaller.exe

#Install WSL - Install Linux Distro for Windows

# https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine
