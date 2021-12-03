# DCBLSTM-PSP: Results #


Results
-------
This directory contains the best model attained from training the DCBLSTM models after rigorous tuning and trial and error. This model achieved a peak Q8 accuracy of X% with the CB513 dataset, whilst also acheiving X% and Y% accuracy with the CASP10 and CASP11 datasets. The model was trained for 100 epochs but due to the EarlyStopping TensorFlow callback, training was halted after ~60 epochs, after the loss metric did not improve any further. The directory structure and each file explanation is discussed further below:

```
results
| └── model_logs
| └── model_checkpoints
│ └── model_plots         
│        └── figure1.png
│        └── figure2.png
|        └── ....png
│ └── model.h5
| └── model.png
│ └── model_history.pckl
│ └── model_arch.json
│ └── model_output.csv
| └── model_config.json
| └── training.log
└-
```

* `/model_logs` - TensorFlow logs.
* `/model_checkpoints` - model checkpoints during training.
* `/model_plots` - metric plots after model evaluation.
* `/model.h5` - saved Keras model.
* `/model.png` - visualisation of the model structure.
* `/model_history.pckl` - pickle of model history during training.
* `/model_arch.json` - model architecture in json format.
* `/model_output.csv` - model output results after training.
* `/model_config.json` - configuration file used to build and train the model.
* `/training.log` - model training logs.
