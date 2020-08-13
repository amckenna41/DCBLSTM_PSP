#check if bucket has been created, if not call aws_resoruces

$BUCKET_NAME = ""
#copy dataset from cwd to s3 bucket

aws s3 cp ./data s3://$BUCKET_NAME/data --recursive
