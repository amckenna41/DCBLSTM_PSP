# protein_structure_prediction_DeepLearning
Secondary Protein Structure Prediction using Neural Networks and Deep Learning. 

Datasets used for training:
cullpdb+profile_6133.npy.gz - this dataset is dividied into training/testing/validation/test sets. 
cullpdb+profile_6133_filtered.npy.gz - this dataset is filtered to remove redundancies with the CB513 test dataset. 

The cullpdf_profile6133 dataset is in numpy format, thus for training and usabilitiy, it is reshaped into a 3-D array of size 6133 x 700 x 57 (Protein x amino acids(peptide chain) x features (for each amino acid)).

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