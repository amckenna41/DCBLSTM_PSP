

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

**How to change GCP Ai-Platform parameters:** <br>

**Google Cloud Platform Architecture**<br>

![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/psp_gcp/gcp_architecture.png?raw=true)
