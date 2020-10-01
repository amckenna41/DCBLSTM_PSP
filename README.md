# protein_structure_prediction_DeepLearning
Secondary Protein Structure Prediction using Neural Networks and Deep Learning.

**status**
> Development Stage

**Protein Structure Prediction**

What is it, ??

##About my model ##
ULSTM vs BLSTM

##Conclusions##

The paper is available at...

Datasets used for training:
cullpdb+profile_6133.npy.gz - this dataset is dividied into training/testing/validation/test sets.
cullpdb+profile_6133_filtered.npy.gz - this dataset is filtered to remove redundancies with the CB513 test dataset.

The cullpdf_profile6133 dataset is in numpy format, thus for training and useabilitiy, it is reshaped into a 3-D array of size 6133 x 700 x 57 (Protein x amino acids(peptide chain) x features (for each amino acid)).

In the used dataset the average protein chain consists of 208 amino acids.

For 8-SP, Alpha-helix is sub-divided into three states: alpha-helix (’H’), 310 helix (’G’) and pi-helix (’I’). Beta-strand is sub-divided into: beta-strand (’E’) and beta-bride (’B’) and coil region is sub-divided into: high curvature loop (’S’), beta-turn (’T’) and irregular (’L’)

The 57 features are:
[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq' - X is used to represent unknown amino acid.
[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
[31,33): N- and C- terminals;
[33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

Among the 57 features, 22 represent the primary structure (20 amino acids, 1 unknown or any amino acid, 1 'No Seq' -padding-), 22 the Protein Profiles (same as primary structure) and 9 are the secondary structure (8 possible states, 1 'No Seq' -padding-).

The corresponding amino acid for the single letter code can be found at:
http://130.88.97.239/bioactivity/aacodefrm.html
And the structure for these amino acids can be found at:
http://130.88.97.239/bioactivity/aastructfrm.html

##
The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence.
[22,31) and [33,35) are hidden during testing.

The 8 different labels for the secondary protein sequence are:

* alpha helix
* beta strand
* loop or irregular
* beta turn
* bend
* 310-helix
* beta bridge
* pi helix

The dataset division for the first cullpdb+profile_6133.npy.gz dataset is
[0,5600) training
[5605,5877) test
[5877,6133) validation

 For the filtered dataset cullpdb+profile_6133_filtered.npy.gz, all proteins can be used for training and test on CB513 dataset.

###

These datasets are available at:
https://www.princeton.edu/~jzthree/datasets/ICML2014/

Datasets used for testing:
cb513+profile_split1.npy.gz
casp10.h5
casp11.h5

The CB513 dataset is available at:
https://www.princeton.edu/~jzthree/datasets/ICML2014/

The CASP10 and CASP11 datasets are available at:
https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d

## Installation - Python Requirements

The required Python modules/packages are in requirements.txt. Call
```
pip3 install -r requirements.txt
```



## Implementation

This PSP project was implemented using the Keras API which is a deep learning API that runs on top of the Tensorflow machine learning framework.
Four main approaches were explored and tested. Firstly, a standalone CNN network, a CNN plus a fully-connected DNN, a recurrent neural network consisting of LSTM layers followed by a DNN and finally a combination of all three of the mentioned components which formed the final model architecture - a CNN followed by a RNN followed by a DNN. More information about the network topologies can be seen below. <br>

1) PSP using CNN

2) PSP using CNN + DNN

3) PSP using RNN + DNN

4) PSP using CNN + RNN + DNN

## Running model locally with default parameters:
```
python main_local.py
```

With the inclusion of the recurrent layers (LSTM), the computational complexity of the network dramatically increases, therefore it is not feasible to build the model using the whole dataset. Thus, a Cloud implementation to run the model successfully was created using the Google Clodu Platform.

##Running model and deploying to GCP:** <br>
Change current working directory to psp_gcp
```
cd psp_gcp
```
To be able to run the model on the cloud you must have an existing GCP account and have the Google Cloud SDK/CLI pre-installed. Follow the gcp_config script in psp_gcp/scripts directory, which contains the relevant commands to execute to configure your GCP account. <br>
Call bash script ./gcp_training.sh on a command line/terminal. This will call the BLSTM_3xConv_Model on the GCP Ai-Platform with the default settings and parameters.


**References**
[1]: https://www.princeton.edu/~jzthree/datasets/ICML2014/
[2]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2940-0
[3]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2280-5
[4]:
**status**
> Development Stage
