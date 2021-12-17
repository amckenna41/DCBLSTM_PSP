# Models used in project

#**Note!** <br>
If running locally, ensure you have sufficient h/w resources to be able to build and train the model locally. Due to the size of the training dataset and the complexity of the models, it is infeasible to run a model locally on an average laptop device. Alternatively, you could run the models on a smaller portion of the dataset and use a large batch size. It was impractical to run the mentioned models locally, therefore a cloud distribution was created that allowed for the building and training of the models using Google Cloud Platform; this can be seen in the psp_gcp directory.
<br>

Bidirectional LSTM model (psp_dcblstm_model)
--------------------------------------------

This model consisted of 3 convolutional layers with Batch Normalisation and Dropout applied for
normalisation and regularisation, respectively. Next, two recurrent layers in the form of bidirectional
LSTM's were implemented to add a recurrent/temporal component to the model. The convolutional components of the network captured local dependancies in the protein sequence with the latter LSTM elements capturing any long-distance dependancies. 3 dense fully-connected layers were implemented that were fed in by the recurrent components with the final dense layer being the 8 node output for the 8 secondary structure labels.

![alt text](https://raw.githubusercontent.com/amckenna41/DCBLSTM_PSP/master/images/model.png)


Unidirectional LSTM model (psp_ulstm_model)
-------------------------------------------

This model consisted of 3 convolutional layers with Batch Normalisation and Dropout applied for
normalisation and regularisation, respectively. Next, three recurrent layers in the form of unidirectional
LSTM's were implemented to add a recurrent/temporal component to the model. The convolutional components of the network captured local dependancies in the protein sequence with the latter LSTM elements capturing any long-distance dependancies. 3 dense fully-connected layers were implemented that were fed in by the recurrent components with the final dense layer being the 8 node output for the 8 secondary structure labels.


Auxillary Models
----------------

These auxiliary models were additionally created for evaluation and testing purposes and assisted in the overall selection of parameters and model configuration for the final DCBLSTM/DCULSTM models.

* `psp_brnn.py` - bidirectional recurrent neural network model.
* `psp_rnn.json` - recurrent neural network model.
* `psp_cnn.json` - convolutional neural network model.
* `psp_dcbgru.json` - bidirectional gated recurrent unit model.
* `psp_dcblstm_3lstm.json` - 3x bidirectional long-short-term-memory model.
* `psp_dcblstm_4conv.json` - 4x convolutional bidirectional long-short-term-memory model.
* `psp_dcblstm_pooling.json` - bidirectional long-short-term-memory with MaxPooling model.
* `psp_dcblstm.json` - bidirectional long-short-term-memory model.
* `psp_dcugru.json` - unidirectional gated recurrent unit model.
* `psp_dculstm_4conv.json` - 4x convolutional unidirectional long-short-term-memory model.
* `psp_dculstm_4lstm.json` - 4x unidirectional long-short-term-memory model.
* `psp_dculstm.json` - unidirectional long-short-term-memory model.
* `psp_dnn.json` - deep neural network model.
* `dummy.json` - dummy model.
