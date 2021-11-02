################################################################################
#######        Cloud Function for notifying results of training          #######
################################################################################

#import required modules and dependancies
# import base64
from google.cloud import storage, exceptions
from googleapiclient import errors
from googleapiclient import discovery
from google.cloud import logging
from google.cloud import pubsub_v1
from google.oauth2 import service_account
import smtplib
from email.message import EmailMessage
import imghdr
import mimetypes
import os
import json
import pandas as pd

def notification_func(event, context):
    """
    Description:
        Main entry function for Google Cloud function that notifes and emails when
        training has completed. Results are parsed and emailed. Triggered from a message
        on a Cloud Pub/Sub topic.
    Args:
        :event (dict): Event payload.
        :context (google.cloud.functions.Context): Metadata for the event.
    Returns:
        None
    """
    job_name = os.environ['JOB_NAME']
    bucket_name = os.environ['BUCKET']
    results_filename = 'output_results.csv'
    blob_path = os.path.join(bucket_name, job_name, results_filename)
    bucket_path = 'gs://' + blob_path

    #Instantiate a Google Cloud Storage client and specify required bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    #read csv from GCP bucket
    data_df = pd.read_csv(bucket_path)

    #save csv locally so it can be attached to the email later
    filepath = '/tmp/' + results_filename
    data_df.to_csv(filepath)

    #call send_email func to parse training results and send via email
    send_email(data_df, filepath)

def send_email(data_df, csv_path):
    """
    Description:
        Parses required training results, packages them and sends them to receipent via
        email using SMTP over SSL.
    Args:
        :data_df (dataframe): training results dataframe.
        :csv_path (str): path to locally stored results csv for sending results via attachment.
    Returns:
        None
    """
    #get server environment variables
    EMAIL_ADDRESS = os.environ['EMAIL_USER'] #to - 45678
    EMAIL_PASS = os.environ['EMAIL_PASS']
    from_mail = os.environ['FROM_MAIL'] #from - 3456
    job_name = os.environ['JOB_NAME']
    bucket_name = os.environ['BUCKET']
    mail_server = 'smtp.gmail.com'
    arch_filename = 'model_architecture.json'
    blob_path = os.path.join(job_name, arch_filename)

    #Instantiate a Google Cloud Storage client and specify required bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    print("Blob path:",blob_path)

    #initialise mail server parameters
    msg = EmailMessage()
    msg['Subject'] = f'Results from: {job_name}'
    msg['From'] = from_mail
    msg['To'] = EMAIL_ADDRESS

    epochs = data_df['Epochs'][0]
    lr = data_df['Learning Rate'][0]
    batch_size = data_df['Batch Size'][0]

    #parse dataframe to get requried results - getting accuracy, precision and recall for training set and all test datasets
    training_acc = data_df['Training Accuracy'][0]
    training_loss = data_df['Training Loss'][0]
    training_recall = data_df['Training Recall'][0]
    training_precision = data_df['Training Precision'][0]

    cb513_acc = data_df['CB513 Evaluation Accuracy'][0]
    cb513_recall = data_df['CB513 Recall'][0]
    cb513_precision = data_df['CB513 Precision'][0]

    casp10_acc = data_df['CASP10 Evaluation Accuracy'][0]
    casp10_recall = data_df['CASP10 Recall'][0]
    casp10_precision = data_df['CASP10 Precision'][0]

    casp11_acc = data_df['CASP11 Evaluation Accuracy'][0]
    casp11_recall = data_df['CASP11 Recall'][0]
    casp11_precision = data_df['CASP11 Precision'][0]

    #create bucket blob
    blob = bucket.blob(blob_path)

    #Download the contents of the blob as a string and then parse it using json.loads() method
    arch_json = json.loads(blob.download_as_string(client=None))
    layers = []

    for i, k in enumerate(arch_json['config']['layers']):
        layers.append(k['class_name'])

    layer_str = ' '.join(layers)

    #email body
    results_str= '''
        Job: {} has completed, the results are below... \n
        Epochs: {} \n
        Learning Rate: {} \n
        Batch Size: {} \n\n
        Results: \n
        Training Accuracy: {} \n
        Training Loss: {} \n
        Training Precision: {} \n
        Training Recall: {} \n\n
        CB513 Test Accuracy: {} \n
        CB513 Recall: {} \n
        CB513 Precision: {} \n\n
        CASP10 Test Accuracy: {} \n
        CASP10 Recall: {} \n
        CASP10 Precision: {} \n\n
        CASP11 Test Accuracy: {} \n
        CASP11 Recall: {} \n
        CASP11 Precision: {} \n
        Model Architecture: \n
        {} \n
        Results stored in: {} bucket \n
    '''.format(job_name, epochs, lr, batch_size,training_acc, training_loss, training_precision,
                training_recall, cb513_acc, cb513_recall, cb513_precision, casp10_acc,
                casp10_recall, casp10_precision, casp11_acc, casp11_recall,
                casp11_precision, layer_str, bucket_name)

    #set body of email message
    msg.set_content(results_str)

    #get filetype of output results file
    ctype, encoding = mimetypes.guess_type(csv_path)

    #set required encodings for attachment csv file
    if ctype is None or encoding is not None:
        # No guess could be made, or the file is encoded (compressed), so
        # use a generic bag-of-bits type.
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)

    #open temp output csv for reading
    with open(csv_path,'rb') as f:
        filename = f.name
        #set ouput csv as email attachment
        msg.add_attachment(f.read(),
                            maintype=maintype,
                            subtype=subtype,
                            filename=filename)

    #open secure SSL connection on email server
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        #login using SMTP and send message to receipient
        smtp.login(from_mail, EMAIL_PASS)
        smtp.send_message(msg)

def get_layer_attributes(layers):
    pass

def check_job_status():
    #parse job logs and email if job fails - send error etc.
    pass

   # msg = MIMEMultipart()
   #  msg['To'] = "4_server_dev@company.com"
   #  msg['From'] = "system@company.com"
   #  msg['Subject'] = "Selenium ClearCore_Regression_Test_Report_Result"
   #
   #  body = MIMEText('Test results attached.', 'html', 'utf-8')
   #  msg.attach(body)  # add message body (text or html)
   #
   #  for f in files:  # add files to the message
   #      file_path = os.path.join(dir_path, f)
   #      attachment = MIMEApplication(open(file_path, "rb").read(), _subtype="txt")
   #      attachment.add_header('Content-Disposition','attachment', filename=f)
   #      msg.attach(attachment)



# def get_job_status(job_name):
#     """
#     Description:
#
#     Args:
#         :job_name(str):
#
#     """
#     # job_logs = subprocess.check_output(["gcloud", "ai-platform","jobs","describe",job_name])
#     job_logs = subprocess.Popen(["gcloud", "ai-platform","jobs","describe",job_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#
#     output, err = job_logs.communicate(b"input data that is passed to subprocess' stdin")
#
#     status=""
#     for item in (output.decode('UTF-8')).split("\n"):
#         if ("state:" in item):
#             status = item.strip()
#             status = (status[status.find(':')+1:]).strip()
#
#     err_message=""
#     if (status=="FAILED"):
#         for item in (output.decode('UTF-8')).split("\n"):
#             if ("errorMessage:" in item):
#                 err_message = item.strip()
#
#     #get err_message down to etag
#     return status, err_message
