#Script for deploying, training and evaluating Keras models via AWS SagemMaker

echo "Current AWS CLI Version: "
(aws --version) || ./scripts/aws_resources.sh
