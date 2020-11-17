# protein_structure_prediction_DeepLearning
========================================
Secondary Protein Structure Prediction using Neural Networks and Deep Learning.
========================================

|ProjectStatus|_ |Version|_ |BuildStatus|_ |Coverage|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/neurodsp.svg
.. _Version: https://pypi.python.org/pypi/neurodsp/

.. |BuildStatus| image:: https://travis-ci.com/neurodsp-tools/neurodsp.svg
.. _BuildStatus: https://travis-ci.com/github/neurodsp-tools/neurodsp

.. |Coverage| image:: https://codecov.io/gh/neurodsp-tools/neurodsp/branch/master/graph/badge.svg
.. _Coverage: https://codecov.io/gh/neurodsp-tools/neurodsp

.. |License| image:: https://img.shields.io/pypi/l/neurodsp.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/neurodsp.svg
.. _PythonVersions: https://pypi.python.org/pypi/neurodsp/

.. |Publication| image:: https://joss.theoj.org/papers/10.21105/joss.01272/status.svg
.. _Publication: https://doi.org/10.21105/joss.01272



**status**
> Development Stage

**Protein Structure Prediction**
Protein Structure Prediction (PSP) is the determination of a protein's structure from its initial primary amino acid sequence. Here we focus on secondary protein structure prediction (SPSP) which acts as an intermediate between the primary and tertiary. PSP is one of the most important goals in the field of bioinformatics and remains highly important in the field of medicine and biotechnology, e.g in drug design.[1] The secondary structure is commonly broken down into 8 categories:

* alpha helix ('H')
* beta strand ('E')
* loop or irregular ('L')
* beta turn ('T')
* bend ('S')
* 3-helix (3-10 helix) ('G')
* beta bridge ('B')
* 5-helix (pi helix) ('I')

Most proteins fall into the category of 4 structures. The primary structure is the , the seondary structure is, tertiary structure is and the quaternary structure is...  ... A visualisation of these structures can be seen below in Figure 1.

<br> bit of info about proteins and their struture...
![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/protein_structure.jpeg?raw=true)

**Approach**

Many different approaches for implementing effective PSP have been proposed which have included Convolutional Neural Nets, SVM's, random forests, KNN, Hidden Markov Models etc. There has also been much recent research with the utilisation of recurrent neural nets, specifically using GRU's (Gated Recurrent Units) and LSTM's (Long-Short-Term-Memory). These recurrent components help map long-distance dependancies in the protein chain, whereby an amino acid may be influenced by a residue much earlier or later in the sequence, this can be attributed to the complex protein folding process. <br>

My model focused on LSTM's, specifically bidirectional LSTM's which allow for the LSTM unit to consider the protein sequence in the forward and backward direction. Additionally, to map local dependancies and context between adjacent residues, a CNN preceded the recurrent component of the model where 1-Dimensional convolutional layers were used.
Optimisation and regularisation techniques were applied to the model to maximise performance and effiency.

##Conclusions and Results##

#insert graphs and tables here ....

The paper is available at...

**Datasets**

Datasets used for training:
cullpdb+profile_6133.npy.gz - this dataset is dividied into training/testing/validation/test sets.
cullpdb+profile_6133_filtered.npy.gz - this dataset is filtered to remove redundancies with the CB513 test dataset.

The cullpdf_profile6133 dataset is in numpy format, thus for training and useabilitiy, it is reshaped into a 3-D array of size 6133 x 700 x 57 (Protein x amino acids(peptide chain) x features (for each amino acid)).

On average a polupeptide chain consists of around 200 - 300 amino acids; in the CullPDB dataset, the average polypeptide chain has 208 amino acids.
The 57 features are:
[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq' - X is used to represent unknown amino acid.
[22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
[31,33): N- and C- terminals;
[33,35): relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
[35,57): protein sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

The corresponding amino acid for the single letter code can be found at:
http://130.88.97.239/bioactivity/aacodefrm.html
And the structure for these amino acids can be found at:
http://130.88.97.239/bioactivity/aastructfrm.html

The dataset division for the first cullpdb+profile_6133.npy.gz dataset is
[0,5600) training
[5605,5877) test
[5877,6133) validation

 For the filtered dataset cullpdb+profile_6133_filtered.npy.gz, all proteins can be used for training and test on CB513 dataset.

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

**Installation - Python Requirements**

The required Python modules/packages are in requirements.txt. Call
```
pip3 install -r requirements.txt
```

**Implementation**

This PSP project was implemented using the Keras API which is a deep learning API that runs on top of the Tensorflow machine learning framework. The model consisted of 3 main components, a 1-Dimensional CNN for capturing local context between adjacent amino acids, a bidirectional LSTM RNN for mapping long distance dependancies within the sequence and a deep fully-connected network used for dimensionality reduction and classification. The design of the model can be seen below:

![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/model_design.png?raw=true)


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
To be able to run the model on the cloud you must have an existing GCP account and have the Google Cloud SDK/CLI pre-installed. Follow the README.md and gcp_config script in psp_gcp directory, which contains the relevant commands and steps to execute to configure your GCP account. <br>
Call bash script ./gcp_training.sh on a command line/terminal. This will call the BLSTM_3xConv_Model on the GCP Ai-Platform with the default settings and parameters.


#user inputs protein sequence, BLAST run on sequence through BLAST API to convert to PSSM? - main input
#protein sequence from fasta is converted into one-hot encoded vector per AA

I am a newcomer to ncbi-blast-2.2.29+. Previously, I had been using blastpgp to gain PSSM. In blastpgp, to gain a PSSM of a protein fasta, I ran command as follows:

blastpgp -a $BLAST_NUM_CPUS -t 1 -i $file -j2 -o $id$chain.nr.blast -d NR -v10000 -b10000 -K1000 -h0.0009 -e0.0009 -C $id$chain.nr.chk -Q $id$chain.nr.pssm.
https://www.biostars.org/p/14253/
https://www.biostars.org/p/2997/

**References**
[1]: https://www.princeton.edu/~jzthree/datasets/ICML2014/
[2]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2940-0
[3]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2280-5
[4]:
**status**
> Development Stage


**add draw.io to images folder
