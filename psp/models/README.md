Models Directory stores all the Python code for all of the created models.
Directory used for building, fitting and evaluating models locally.
<br>
The project focused on the evaluation of unidirectional and bidirectional LSTM's (ULSTM/BLSTM).
Current model hyperparameters are ones that have proved to give the greatest accuracy and loss metrics.

#Model Design
![alt text](https://github.com/amckenna41/protein_structure_prediction_DeepLearning/blob/master/images/model_design.png?raw=true)

**Note!** <br>
If running locally, ensure to use a small portion of the training dataset as well as large batch_size due to the complexity of the networks. If the full dataset is required to train the models then consider using the GCP cloud implementations in the psp_gcp directory.

#Bidirectional LSTM model:

```
psp_blstm_model
```

#Unidirectional LSTM model:

```
psp_ulstm_model
```

#Data for plotting model metrics

```
plot_model
```

#Run Tests: <br>

```
python3 test_model
```
