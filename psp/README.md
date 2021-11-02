# Protein Structure Prediction




## Conclusions and Results

#insert graphs and tables here ....

## Datasets

Training:
cullpdb+profile_6133.npy.gz - this dataset is divided into training/testing/validation/test sets.
cullpdb+profile_6133_filtered.npy.gz - this dataset is filtered to remove redundancies with the CB513 test dataset.

The cullPDB dataset is reshaped into a 3-D array of size 6133 x 700 x 57 (Protein x amino acids(peptide chain) x features (for each amino acid)). In the dataset, the average polypeptide chain has 208 amino acids.<br>

**The 57 features are:**
[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq' - X is used to represent unknown amino acid.
[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
[31,33): N- and C- terminals;
[33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
[35,57): protein sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

The corresponding amino acid for the single letter code can be found at:
http://130.88.97.239/bioactivity/aacodefrm.html  <br>
And the structure for these amino acids can be found at:
http://130.88.97.239/bioactivity/aastructfrm.html

The dataset division for the first cullpdb+profile_6133.npy.gz dataset is
[0,5600) training <br>
[5605,5877) test  <br>
[5877,6133) validation <br>

These datasets are available at:
https://www.princeton.edu/~jzthree/datasets/ICML2014/

Testing:
- cb513+profile_split1.npy.gz
- casp10.h5
- casp11.h5

The CB513 dataset is available at:
https://www.princeton.edu/~jzthree/datasets/ICML2014/

The CASP10 and CASP11 datasets are available at:
https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d


## Implementation

This PSP project was implemented using the Keras API which is a deep learning API that runs on top of the TensorFlow machine learning framework. The model consisted of 3 main components, a 1-Dimensional CNN for capturing local context between adjacent amino acids, a bidirectional LSTM RNN for mapping long distance dependancies within the sequence and a deep fully-connected network used for dimensionality reduction and classification. The design of the model can be seen below:

<img src="https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/model_design.png" height="400" width="250">



## System Requirements:
```
Python3
hw requirements **
```

## Running model locally with default parameters:

**From main repo dir, change to PSP directory:**
```
cd psp
```

**Create Python virtual environment:**
```
python3 -m venv psp_venv
source psp_venv/bin/activate

```
**The required Python modules/packages can be installed by:**
```

pip install -r requirements.txt

```

**Run train function to build and train model:**
```

python train.py
-batch_size : training dataset batch size
-logs_dir : Directory for TensorBoard logs generated from model

```

## Show TensorBoard Logs....
$ cd ~/Desktop/tensorboard

tensorboard --logdir=$LOGS_DIR --port=8080

--logdir='saved/models/logs'

## Cloud Distribution

With the inclusion of the recurrent layers (LSTM), the computational complexity of the network dramatically increases, therefore it is not feasible to build the model using the whole dataset locally due to the computational constraints. Thus, a Cloud implementation to run the model successfully was created using the Google Cloud Platform.

**Change current working directory to psp_gcp:**
```
cd psp_gcp
```

To be able to run the model on the cloud you must have an existing GCP account and have the Google Cloud SDK/CLI pre-installed. Follow the README.md and in the psp_gcp directory, which contains the relevant commands and steps to follow to configure your GCP account. <br>

**From a cmd line/terminal, to train current CDBLSTM model configuration:**
```
./gcp_training.sh

-b batch size
-e epochs
-ad proportion of training dataset to use
-m module - python module containing commands to build and train model

```
# Model Saving Structure

* all model attributes and objects are stored in the '/saved_models' folder
* when training a model, a new folder within the '/saved_models' folder will be created. This folder will be named according to - 'model-name + current-date/time.'  
* model visualisations and plots will be stored within this newly created models folder in a folder called 'plots'
- e.g

```
psp
├── saved_models
│   └── psp_cdblstm_model_YYYY-MM-DD_HH:MM
|                       └── logs
|                       └── checkpoints
│                       └── plots
│                             └── figure1.png
│                             └── figure2.png
│                       └── model.h5
│                       └── model_history.pckl
│                       └── model_arch.json
│                       └── model_output.csv
│   └── psp_cdulstm_model_YYYY-MM-DD_HH:MM
|                       └── logs
|                       └── checkpoints
│                       └── plots
│                             └── figure1.png
│                             └── figure2.png
│                       └── model.h5
|                       └── model.png
│                       └── model_history.pckl
│                       └── model_output.csv
|                       └── training.log
└-


https://gist.githubusercontent.com/ryanflorence/daafb1e3cb8ad740b346/raw/37fc2af6e55acc2b2fbf2aea23fec5f5c48e2fc5/folder-structure.md
```

## Directory folders:

* `/data` - downloads and loads in the required training and test datasets
* `/models` - stores required Keras code to build required models, also plots the results and metrics from the models after training (plot_model)
* `/tests` - tests for datasets and models
