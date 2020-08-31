Data directory stores the datasets required to train the models locally.<br> <br>
**get_dataset.py:**
Downloads training and test datasets used in the models and stores them in data dir.<br> <br>
**load_dataset.py:**
Unzips, formats and reshapes training and test datasets, which is required prior to creation of the neural network models.  
<br> <br>
From a terminal or command prompt, calling the load_dataset.py will download the training and all the test datasets used in this project. Call: python load_dataset from /data directory
or python -m data.load_dataset from root of project.
<br <br>

**Datsets used in project** <br>
Training Dataset: Cullpdb profile filtered  <br>

This dataset consists of 5534 proteins and is a subset of the original Cullpdb 6133 dataset, with some protein sequences removed to remove any redundancies with the CB513 test dataset. Using this filtered dataset meant all proteins can be used for training and tested on CB513 dataset.The dataset is reshaped into 3 dimensions of 5534 proteins x 700 amino acids x 57 features. Thus, the 700 amino acids represent the peptide chain of the protein, with each of the amino acids in the chain having 57 features. <br>
The dataset was split into training and validation datasets, with a split of 5278 proteins for training and 256 for validation.

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz

Primary Test Dataset: CB513 <br>

Available from:
http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz


Other Test Datsets: CASP10, CASP11 <br>

These 2 auxillary test datasets were taken from the 10th and 11th iteration of the CASP ( ) competition
Available from:

https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp10.h5
https://github.com/amckenna41/protein_structure_prediction_DeepLearning/raw/master/data/casp11.h5
