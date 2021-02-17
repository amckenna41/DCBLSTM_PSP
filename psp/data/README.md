
# Datasets used in project

## Training Datasets: cullpdb+profile_filtered  <br>

The datasets used in
* cullpdb+profile_6133 --
* cullpdb+profile_6133_filtered --
* cullpdb+profile_5926 --
* cullpdb+profile_5926_filtered --

This dataset consists of 5534 proteins and is a subset of the original Cullpdb 6133 dataset, with some protein sequences removed to remove any redundancies with the CB513 test dataset. Using this filtered dataset meant all proteins can be used for training and tested on CB513 dataset.The dataset is reshaped into 3 dimensions of 5534 proteins x 700 amino acids x 57 features. Thus, the 700 amino acids represent the peptide chain of the protein, with each of the amino acids in the chain having 57 features. <br>
The dataset was split into training and validation datasets, with a split of 5278 proteins for training and 256 for validation.

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz

**CullPDB dimensions:**
> 5278, 700, 21



Update 2018-10-28:
The original 'cullpdb+profile_6133.npy.gz' and 'cullpdb+profile_6133_filtered.npy.gz' files uploaded contain duplicates. The fixed files with duplicates removed are  'cullpdb+profile_5926.npy.gz' and 'cullpdb+profile_5926_filtered.npy.gz'.

The corresponding dataset division for the cullpdb+profile_5926.npy.gz dataset is
[0,5430) training
[5435,5690) test
[5690,5926) validation



## CB513 test datasets:

CB513 is made up of 514 protein sequences and 87,041 total residues. As there exists some redundancy between CB513 and CB6133, CB6133 is filtered by removing sequences having over 25% sequence similarity with sequences in CB513. After filtering, 5534 proteins left in CB6133 are used as training samples.

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

**CB513 dimensions:**
> 514, 700, 21


## CASP10 test dataset:

This additional test dataset was taken from the 10th iteration of the biennial CASP (Critical Assessment of Structure Prediction) competition. CASP10 contains 123 domain sequences extracted from 103 chains. There are 22,041 residues for each protein in the sequences of CASP10.

Available from:

https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5

**CASP10 dimensions:**
> 123, 700, 21

## CASP11 test dataset:

This dataset was taken from the 11th iteration of the biennial CASP (Critical Assessment of Structure Prediction) competition. CASP11 contains 105 domain sequences extracted from 85 chains. There are 20,498 residues for each protein in the sequences of CASP10.

Available from:

https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5

**CASP11 dimensions:**
> 105, 700, 21



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
