# Protein Structure Prediction

Datasets
--------

An in-depth description of the training and test datasets used in this project was described in the
readme of the data directory on the repo: https://github.com/amckenna41/DCBLSTM_PSP/tree/master/data
<!--
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
https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d -->

<!--
## Implementation

This PSP project was implemented using the Keras API which is a deep learning API that runs on top of the TensorFlow machine learning framework. The model consisted of 3 main components, a 1-Dimensional CNN for capturing local context between adjacent amino acids, a bidirectional LSTM RNN for mapping long distance dependancies within the sequence and a deep fully-connected network used for dimensionality reduction and classification. The design of the model can be seen below:

<img src="https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/model_design.png" height="400" width="250"> -->
<!--

## Conclusions and Results

#insert graphs and tables here ....


## System Requirements:
```
Python3
hw requirements **
``` -->

**Create Python virtual environment:**
```
python3 -m venv psp_venv
source psp_venv/bin/activate
```
**The required Python modules/packages can be installed by:**
```
pip install -r requirements.txt
```

**Run main function to build and train model with specified json config file, e.g running dummy model using dummy.json:**
```
python main.py --config=dummy

--config: configuration json file
```

Model Saving Structure
----------------------

The code pipeline created compiles all of the training assets and logs into one output folder named using the model name with the current date/time appended to it. Below is the structure of that output folder

```
output_folder
├── job_name_DDMMYYY:MM
|   └── model_logs
|   └── model_checkpoints
│   └── model_plots         
│         └── figure1.png
│         └── figure2.png
|         └── ....png
│   └── model.h5
|   └── model.png
│   └── model_history.pckl
│   └── model_arch.json
│   └── model_output.csv
|   └── model_config.json
|   └── training.log
└-
```

Directory folders
-----------------

* `/models` - python scripts for main and auxiliary models used in project.
