#Shell script for making all the resources required for deployment on AWS

BUCKET_NAME="BUCKET_NAME"
REGION_NAME="REGION_NAME"

echo "Current AWS CLI Version: "
(aws --version) || ./scripts/aws_config.sh

#get current user details
#print out current config
#check if any configuration requirements required for connection to SagemMaker


#check if bucket already exists, if not then make bucket
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
echo "Bucket already exists" else
aws mb s3://$BUCKET_NAME --region $REGION_NAME fi #--acl public-read (set acl for bucket)

#get file size of bucket
aws s3 ls s3://$BUCKET_NAME --recursive | awk 'BEGIN {total=0}{total+=$3}END{print total/1024/1024" MB"}'

#list buckets
aws s3 ls s3://$BUCKET_NAME

#create models folder
aws s3api put-object --bucket $BUCKET_NAME --key models/

#create checkpoints folder
aws s3api put-object --bucket $BUCKET_NAME --key checkpoints/

#get bucket policy
aws s3api get-bucket-policy --bucket $BUCKET_NAME

#get and put bucket policy from policy.json file in cwd
aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://policy.json

#get bucket access control list
aws s3api get-bucket-acl --bucket $BUCKET_NAME

#change bucket access control list
# aws s3api put-bucket-acl --bucket $BUCKET_NAME --grant-full-control emailaddress=user1@example.com,emailaddress=user2@example.com --grant-read uri=http://acs.amazonaws.com/groups/global/AllUsers

#Sagemaker setup
