# Models used in project

<br>
The project focused on the evaluation of unidirectional and bidirectional LSTM's (ULSTM/BLSTM).
Current model hyperparameters are ones that have proved to give the greatest accuracy and loss metrics.

#Model Design
![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/model_design.png?raw=true)

#**Note!** <br>
If running locally, ensure you have sufficient hardware to be able to build and train the model locally. Due to the size of the training dataset and the complexity of the models, it is infeasible to run a model locally on an average laptop device. Alternatively, you could run the models on a smaller portion of the dataset and use a large batch size. It was impractical to run the mentioned models locally, therefore a cloud distribution was created that allowed for the building and training of the models using Google Cloud Platform; this can be seen in the psp_gcp directory.
<br>


## Bidirectional LSTM model:

```
psp_blstm_model
```

## Unidirectional LSTM model:

```
psp_ulstm_model
```

# Data for plotting model metrics

```
plot_model
```
