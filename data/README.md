
**Datsets used in project** <br>
Training Dataset: Cullpdb profile filtered  <br>

This dataset consists of 5534 proteins and is a subset of the original Cullpdb 6133 dataset, with some protein sequences removed to remove any redundancies with the CB513 test dataset. Using this filtered dataset meant all proteins can be used for training and tested on CB513 dataset.The dataset is reshaped into 3 dimensions of 5534 proteins x 700 amino acids x 57 features. Thus, the 700 amino acids represent the peptide chain of the protein, with each of the amino acids in the chain having 57 features. <br>
The dataset was split into training and validation datasets, with a split of 5278 proteins for training and 256 for validation.

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz

CullPDB dimensions:
5278, 700, 21


**Primary Test Dataset: CB513** <br>

CB513 is made up of ...

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

CB513 dimensions:
514, 700, 21


**Other Test Datsets: CASP10, CASP11** <br>

These 2 auxillary test datasets were taken from the 10th and 11th iteration of the CASP (Critical Assessment of Structure Prediction) competition.

Available from:

https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5
https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5

CASP10 dimensions:
123, 700, 21

CASP11 dimensions:
105, 700, 21

To download and load data locally into data directory, from data dir and a terminal, call:
```
python load_dataset.py
```

Or from the root of the project, from a terinal, call:
```
python -m data.load_dataset
```

##Run Tests:** <br>

```
python3 test_dataset 
```
