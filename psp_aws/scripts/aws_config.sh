#Script for configuring connection to the AWS and AWS SDK to be able to train and evaluate
#models on the cloud

#Download AWS CLI
# curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg" - Download and install through curl command
# sudo installer -pkg AWSCLIV2.pkg -target /

#Download and install via pip
if !(pip --version == '20.2.1')
then
  echo "Updating pip to latest version"
  sudo -H pip3 install --upgrade pip
else
  echo "Pip up-to-date"
fi

pip install awscli
echo "AWS CLI downloaded"
pip install boto3
echo "Boto3 library downloaded"

#verify installation
which aws
aws --version

# #configure and setup access to AWS resources
# #access key ID and secret access key found in credentials csv created within IAM of AWS console
aws configure

#view current profile configuration setup
aws configure list [--profile profile-name]
