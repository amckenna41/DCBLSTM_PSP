
from datetime import date
import time
import os
from datetime import datetime
from google.cloud import storage

#initialise bucket and GCP storage client

BUCKET_NAME = "keras-python-models-2"
BUCKET_PATH = "gs://keras-python-models-2"

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
# storage_client = storage.Client.from_service_account_json("service-account.json")
#credentials = GoogleCredentials.get_application_default()
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

current_datetime = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

model_output = {}

plots_path = ""

#Saving pickle of history so that it can later be used for visualisation of the model
history_filepath = 'history_' + current_datetime +'.pckl'


#initialise figure filenames
accuracy_fig_filename = 'accuracy_fig'+ current_datetime + '.png'
accuracy_loss_fig_filename = 'accuracy_loss_fig'+ current_datetime + '.png'
loss_fig_filename = 'loss_fig'+ current_datetime + '.png'
mae_fig_filename = 'mae _fig'+ current_datetime + '.png'
mse_fig_filename = 'mse_fig'+ current_datetime + '.png'
recall_fig_filename = 'recall_fig'+ current_datetime + '.png'
precision_fig_filename = 'precision_fig'+ current_datetime + '.png'
precision_recall_fig_filename = 'precision_recall_fig' + current_datetime + '.png'

#initialise filenames for boxplots
accuracy_box_filename = 'accuracy_boxplot_'+ current_datetime + '.png'
loss_box_filename = 'loss_boxplot_'+ current_datetime + '.png'
mae_box_filename = 'mae_boxplot_'+ current_datetime + '.png'
mse_box_filename = 'mse_boxplot_'+ current_datetime + '.png'

#initialise histogram figure names
accuracy_hist_filename = 'accuracy_hist'+ current_datetime + '.png'
loss_hist_filename = 'loss_hist'+ current_datetime + '.png'
mae_hist_filename = 'mae_hist'+ current_datetime + '.png'
mse_hist_filename = 'mse_hist'+ current_datetime + '.png'

#initialise KDE figure names
accuracy_kde_filename = 'accuracy_kde_'+ current_datetime + '.png'
loss_kde_filename = 'loss_kde_'+ current_datetime + '.png'
mae_kde_filename = 'mae_kde_'+ current_datetime + '.png'
mse_kde_filename = 'mse_kde_'+ current_datetime + '.png'

#create empty np arrays of shape (1, ) ?
# history_accuracy_array
# history_val_accuracy_array
#
# history_loss_array
# history_val_loss_array
#
# history_mse_array
# history_val_mse_array
#
# history_mae_array
# history_val_mae_array
#
# history_recall_array
# history_val_recall_array
#
# history_precision_array
# history_val_precision_array
#
