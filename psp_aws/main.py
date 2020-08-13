#Main file for executing and running keras PSP model on AWS Sagemaker

import boto3, re
from sagemaker import get_execution_role
import keras
from keras.models import model_from_json
import tensorflow as tf
import shutil
role = get_execution_role()

#create model
#upload model to S3 bucket
#get model and load from s3 bucket
#call and run script that runs model on Sagemaker



if __name__ == 'main':
    main()
