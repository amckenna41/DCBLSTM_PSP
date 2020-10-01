

**Building, fitting and evaluating the models on Google Cloud Platform** <br>

**Setup and configuration to GCP:** <br>
1.)
Using a terminal/command line, ensure that the current working directory is psp_gcp - cd psp_gcp  <br>
##include picture of setup on GCP console
1.)Execute the gcp_config.sh bash script which installs all the relevant dependancies and libraries required for connection to the GCP from the command line. This script ... <br>

To call the model with the optimum parameters, from a command line, run:
```
./gcp_training
```

To call the hyperparameter tuning script, from a command line call:
```
./gcp_hptuning
```
If you want to change any of the default hyperparameters then pass the parameter in when calling the script, e.g:
```
./gcp_hptuning -epochs 10 -batch_size 42 -alldata 0.5
```

<br>
2.) Execute





**How to change model hyperparameters:** <br>
The model hyperparameters are passed into the main calling module from a bash script - gcp_hptuning.sh. The current parameters in the script are the pre-determined optimal parameters for the model.

**How to change GCP Ai-Platform configuration:** <br>
The configuration  ... can be found in gcp_training_config.yaml . For running the main CDBLSTM/CDULSTM models the high memory CPU n1-highmem-8 machine is sufficent. (run on compute engine)
GPU's and TPU's are also available which were tested with the models but ultimately gave similar results to using just CPU's but at a greater cost, therefore high memory CPU machines were utilised.

More info about the different GCP machine types can be found at:
https://cloud.google.com/compute/docs/machine-types


**Google Cloud Platform Architecture**<br>
The cloud architecture used within the GCP for this project can be seen below were several services were taken advantage of including: Ai-Platform, Compute Engine GCS, Logging, Monitoring and IAM.

![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/gcp_architecture.png?raw=true)


**Running on Windows**
1.) Download and install Google Cloud SDK, following the steps from the guide: https://cloud.google.com/sdk/docs/install#windows
2.) SDK Installation should take you through the authentication process, if not then call - gcloud auth login - from terminal to authenticate account.
3.) From Windows Powershell or terminal call the script gcp_resources.sh to create all the neccessary cloud resources required for the project.
4.) From a Windows Powershell or terminal, call the scripts ./gcp_training.sh to build and deploy default best CDBLSTM model with its optimal parameters. Or call the script ./gcp_hptuning.sh to do hyperparameter tuning.

**Running on Mac/Linux**

1.) Download and install Google Cloud SDK, following the steps from the guide: https://cloud.google.com/sdk/docs/install
2.) SDK Installation should take you through the authentication process, if not then call - gcloud auth login - from terminal to authenticate account.
3.) From terminal call the script gcp_resources.sh to create all the neccessary cloud resources required for the project.
4.) From a terminal, call the scripts ./gcp_training.sh to build and deploy default best CDBLSTM model with its optimal parameters. Or call the script ./gcp_hptuning.sh to do hyperparameter tuning.
