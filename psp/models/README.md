# Models used in project

<br>

#Model Design
![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/model_design.png?raw=true)

#**Note!** <br>
If running locally, ensure you have sufficient h/w resources to be able to build and train the model locally. Due to the size of the training dataset and the complexity of the models, it is infeasible to run a model locally on an average laptop device. Alternatively, you could run the models on a smaller portion of the dataset and use a large batch size. It was impractical to run the mentioned models locally, therefore a cloud distribution was created that allowed for the building and training of the models using Google Cloud Platform; this can be seen in the psp_gcp directory.
<br>

## Bidirectional LSTM model:

```
psp_dcblstm_model
```

## Unidirectional LSTM model:

```
psp_dculstm_model
```

## Auxillary Models:

These auxiillay models were additionally created for evaluation and testing purposes and assisted in the overall selection of parameters and model configuration for the final DCBLSTM/DCULSTM models.
```
psp_dcbgru_model
psp_dcugru_model
psp_cnn_model
psp_dnn_model
psp_rbm_model
psp_rnn_model

```
