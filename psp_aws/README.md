Directory for training and running Keras models via AWS
/scripts folder contains all neccessary scripts required to setup and configure connection to AWS via the AWS SDK and CLI. aws_config script configures the neccessary connection to AWS including downloading and installing the CLI and boto3 python library. aws_resources script creates all the neccessary resources required for training models including an S3 bucket to store the models and logs etc. aws_deploy script actually executes and trains the model on AWS via SagemMaker.

Best way to do this...
From console:
Create Sagemaker Notebook Instance with relevant roles/permissions.
Connect Notebook Instance to github repository folder of psp_aws to get code from cwd
Run code from Notebook instance

https://towardsdatascience.com/run-amazon-sagemaker-notebook-locally-with-docker-container-8dcc36d8524a
https://github.com/qtangs/sagemaker-notebook-container
