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
Primary Test Dataset: CB513 <br>
Other Test Datsets: CASP10, CASP11 <br> 
