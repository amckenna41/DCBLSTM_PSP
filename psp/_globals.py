################################################################################
##########                  Global Variables                          ##########
################################################################################

from datetime import date
from datetime import datetime

#######################################################################
###### Get current date and time, store in current_datetime var #######

current_datetime = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

#####################################################
######      Directories for output and data   #######
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

##################################################
###### Output dictionary for model metrics #######

model_output = {}

#####################################################
###### Filenames for plots and visualisations #######

accuracy_fig_filename = 'accuracy_fig'+ current_datetime + '.png'
accuracy_loss_fig_filename = 'accuracy_loss_fig'+ current_datetime + '.png'
loss_fig_filename = 'loss_fig'+ current_datetime + '.png'
mae_fig_filename = 'mae _fig'+ current_datetime + '.png'
rmse_fig_filename = 'rmse_fig'+ current_datetime + '.png'
mse_fig_filename = 'mse_fig'+ current_datetime + '.png'
recall_fig_filename = 'recall_fig'+ current_datetime + '.png'
precision_fig_filename = 'precision_fig'+ current_datetime + '.png'
precision_recall_fig_filename = 'precision_recall_fig' + current_datetime + '.png'

#initialise filenames for boxplots
accuracy_box_filename = 'accuracy_boxplot_'+ current_datetime + '.png'
loss_box_filename = 'loss_boxplot_'+ current_datetime + '.png'
mae_box_filename = 'mae_boxplot_'+ current_datetime + '.png'
mse_box_filename = 'mse_boxplot_'+ current_datetime + '.png'
rmse_box_filename = 'rmse_boxplot_'+ current_datetime + '.png'

#initialise histogram figure names
accuracy_hist_filename = 'accuracy_hist'+ current_datetime + '.png'
loss_hist_filename = 'loss_hist'+ current_datetime + '.png'
mae_hist_filename = 'mae_hist'+ current_datetime + '.png'
mse_hist_filename = 'mse_hist'+ current_datetime + '.png'
rmse_hist_filename = 'rmse_hist'+ current_datetime + '.png'

#initialise KDE figure names
accuracy_kde_filename = 'accuracy_kde_'+ current_datetime + '.png'
loss_kde_filename = 'loss_kde_'+ current_datetime + '.png'
mae_kde_filename = 'mae_kde_'+ current_datetime + '.png'
mse_kde_filename = 'mse_kde_'+ current_datetime + '.png'
rmse_kde_filename = 'rmse_kde_'+ current_datetime + '.png'

##################################################
