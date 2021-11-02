
# Datasets used in project

#Training Datasets: <br>

* cullpdb+profile_6133  [1]
* cullpdb+profile_6133_filtered
* cullpdb+profile_5926
* cullpdb+profile_5926_filtered

These 4 datasets are variations of the same dataset. The dataset consists of 5534 proteins and is a subset of the original Cullpdb 6133 dataset, with some protein sequences removed to remove any redundancies with the CB513 test dataset. Using this filtered dataset meant all proteins can be used for training and tested on CB513 dataset.The dataset is reshaped into 3 dimensions of 5534 proteins x 700 amino acids x 57 features. Thus, the 700 amino acids represent the peptide chain of the protein, with each of the amino acids in the chain having 57 features. <br>
The dataset was split  <br>
The datasets are downloaded and imported via the load_dataset.py module and loaded via the numpy library where they are split into training, test and validation datasets.

Dataset split for training data cullpdb+profile_5926.npy:
[0,5430) training
[5435,5690) test
[5690,5926) validation



|         Dataset         |    Training   |     Test    |  Validation |
| ----------------------- |:-------------:|:-----------:| -----------:|
| cullpdb+profile_5926.npy|    [0,5430]   | [5435,5690] | [5690.5926] |
| cullpdb+profile_6133.npy|    [0,]

<em>Update 2018-10-28</em>:
The original 'cullpdb+profile_6133.npy.gz' and 'cullpdb+profile_6133_filtered.npy.gz' files uploaded contain duplicates. The fixed files with duplicates removed are  'cullpdb+profile_5926.npy.gz' and 'cullpdb+profile_5926_filtered.npy.gz'.


The training datasets are available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/



#Test datasets:

* CB513+profile_split1.npy [2]
* casp10.h5 [3]
* casp11.h5 [4]

## CB513

CB513 is made up of 514 protein sequences and 87,041 total residues. As there exists some redundancy between CB513 and CB6133, CB6133 is filtered by removing sequences having over 25% sequence similarity with sequences in CB513. After filtering, 5534 proteins left in CB6133 are used as training samples.

**CASP10 dimensions:**
> 514, 700, 21

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

## CASP10

This additional test dataset was taken from the 10th iteration of the biennial CASP (Critical Assessment of Structure Prediction) competition. CASP10 contains 123 domain sequences extracted from 103 chains. There are 22,041 residues for each protein in the sequences of CASP10.

**CASP10 dimensions:**
> 123, 700, 21

Available from:
https://github.com/amckenna41/DCBLSTM/raw/master/data/casp10.h5

## CASP11 :

This dataset was taken from the 11th iteration of the biennial CASP (Critical Assessment of Structure Prediction) competition. CASP11 contains 105 domain sequences extracted from 85 chains. There are 20,498 residues for each protein in the sequences of CASP10.

**CASP11 dimensions:**
> 105, 700, 21

Available from:
https://github.com/amckenna41/DCBLSTM/raw/master/data/casp11.h5

References
----------
\[1\]: <br>
\[2\]: <br>
\[3\]: <br>
\[4\]: <br>
