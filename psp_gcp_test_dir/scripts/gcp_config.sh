#Script for configuring connection to the Google Cloud to be able to train and evaluate
#model in GCP

#1: Install Google Cloud SDK
#install SDK in interactive mode
# curl https://sdk.cloud.google.com | bash

#install sdk in non-interactive mode, save install script locally
# curl https://sdk.cloud.google.com > install.sh
# bash install.sh --disable-prompts

#Authorizing with user account
#authenticate GCP SDK, credentials will be automatically stored in config file
#where the SDK looks for the relevant credentials when using sdk tools
# gcloud auth login

#Authenticate using a service account
# gcloud auth activate-service-account --key-file service-account.json [KEY_FILE_PATH]
# gcloud auth activate-service-account [ACCOUNT] --key-file=[KEY_FILE_PATH]
# # gcloud iam service-accounts keys create - create key for existing service account, save to cwd

#list accounts with credentials on the current system
gcloud auth list

#crrate configuration
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
gcloud config set project [PROJECT_NAME]

#set region in configuration
gcloud config set compute/zone [REGION_NAME]
#view properties of current active configuration
gcloud config list

#list all current configurations
gcloud config configurations list

#delete configuration
gcloud config configurations delete [NAME]

#get current version of sdk
gcloud version
#configuring SDK behind proxy/firewall - https://cloud.google.com/sdk/docs/proxy-settings
# gcloud auth application-default login
#initialise gcloud sd

# gcloud init
# gcloud auth activate-service-account
#gcloud auth application-default login


#Full step-by-step setup and configuration for the GCP and SDK can be found in the docs:
#https://cloud.google.com/sdk/docs/
