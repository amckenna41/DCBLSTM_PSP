===================================================================================
==Secondary Protein Structure Prediction using Machine learning and Deep Learning==
===================================================================================

## status
> Development Stage

## Protein Structure Prediction
Protein Structure Prediction (PSP) is the determination of a protein's structure from its initial primary amino acid sequence. Here we focus on secondary protein structure prediction (SPSP) which acts as an intermediate between the primary and tertiary. PSP is one of the most important goals in the field of bioinformatics and remains highly important in the field of medicine and biotechnology, e.g in drug design and environmental stability [1, 2]. The secondary structure is commonly broken down into 8 categories:

* alpha helix ('H')
* beta strand ('E')
* loop or irregular ('L')
* beta turn ('T')
* bend ('S')
* 3-helix (3-10 helix) ('G')
* beta bridge ('B')
* 5-helix (pi helix) ('I')

Proteins are made up of one or more polypeptide chains of amino acid residues. The constituent amino acids are bonded together by peptide bonds. Proteins have a variety of roles within organisms including enzymes, cell signalling and ligand binding, immune response through antibodies and the various roles fulfilled via structural proteins [3]. Most proteins fall into the category of 4 structures. The primary structure is simply the sequence of amino acids, the secondary structure is recurring arrangements of adjacent amino acids in a polypeptide chain, tertiary structure is the 3-dimensional representation of a protein consisting of a polypeptide chain/backbone with 1 or more secondary protein structures[4], quaternary structure is when a protein consists of more than one polypeptide chain [5]. A visualisation of these structures can be seen below in Figure 1.

<br>
<img src="https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/protein_structure.jpeg" height="400" width="250">

## Approach

Many different machine learning approaches for implementing effective PSP have been proposed which have included Convolutional Neural Nets, SVM's, random forests, KNN, Hidden Markov Models etc [6, 7, 8, 9, 10]. There has also been much recent research with the utilisation of recurrent neural nets, specifically using GRU's (Gated Recurrent Units) and LSTM's (Long-Short-Term-Memory) [11, 12]. These recurrent components help map long-distance dependancies in the protein chain, whereby an amino acid may be influenced by a residue much earlier or later in the sequence, this can be attributed to the complex protein folding process. An LSTM cell is made up of 3 gates - input, output and forget [13]. The forget gate decodes what information the should 'forget' or not. The input gate updates the cell state and output gate controls the extent to which a value in the cell is used to compute the output activation of the LSTM. <br>

![alt text](https://github.com/amckenna41/CDBLSTM_PSP/blob/master/images/lstm_cell.png?raw=true)

Bidirectional LSTM's which allow for the LSTM unit to consider the protein sequence in the forward and backward direction. Additionally, to map local dependancies and context between adjacent residues, a CNN preceded the recurrent component of the model where 1-Dimensional convolutional layers were used.
Optimisation and regularisation techniques were applied to the model to maximise performance and efficiency.

## Conclusions and Results

#insert graphs and tables here ....

The paper is available at...

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

python main_local.py
```



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

## Directory folders:

* `/images` - images used for README
* `/psp` - main protein structure directory containing all modules and code required for building and training models locally.  
* `/psp_gcp` - Google Cloud Platform distribution for training and building models for PSP on the cloud

## References:
1. https://www.princeton.edu/~jzthree/datasets/ICML2014/
2. https://www.sciencedirect.com/science/article/abs/pii/0958166994900264
3. https://www.ncbi.nlm.nih.gov/books/NBK26911
4. https://scholar.google.comscholar_lookup?title=Proteins+and+enzymes.+Lane+Medical+Lectures,+Stanford+University+Publications,+University+Series,+Medical+Sciences&author=KU+Linderstr%C3%B8m-Lang&publication_year=1952&
5. https://pubmed.ncbi.nlm.nih.gov/19059267/
6. https://doi.org/10.1038/srep18962
7. http://airccse.org/journal/ijsc/papers/2112ijsc06.pdf
8. https://doi.org/10.1186/s12859-020-3383-3
9. https://www.sciencedirect.com/science/article/abs/pii/S0022283683714646
10. https://doi.org/10.1093/bioinformatics/9.2.141
11. https://doi.org/10.1093/bioinformatics/btx218
12. https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735
13. https://digital-library.theiet.org/content/conferences/10.1049/cp_19991218

## status
> Development Stage
