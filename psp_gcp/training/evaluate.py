import numpy as np
from keras import backend as K
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from training.training_utils.get_dataset import *
from training.training_utils.plot_model import *
from training.training_utils.globals import *

#maybe create tf summary

'''
Evaluate model using test datasets
'''
def evaluate_model(model, test_dataset="all"):

    if test_dataset.lower() == "cb513":
        evaluate_cb513(model)
    elif test_dataset.lower() == "casp10":
        evaluate_casp10(model)
    elif test_dataset.lower() == "casp11":
        evaluate_casp11(model)
    else:
        evaluate_cb513(model)
        evaluate_casp10(model)
        evaluate_casp11(model)

'''
Evaluate model using CB513 test dataset
'''
def evaluate_cb513(model):

    #load test dataset
    cb513 = CB513()
    # test_hot, testpssm, testlabel = load_cb513()

    print('Evaluating model using CB513 dataset')
    # score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,batch_size=1)
    score = model.evaluate({'main_input': cb513.test_hot, 'aux_input': cb513.testpssm},{'main_output': cb513.testlabel},verbose=1,batch_size=1)

    print('CB513 Evaluation Loss : ', score[0])
    print('CB513 Evaluation Accuracy : ', score[1])

    # pred_casp = model.predict([testpssm, test_hot], verbose=1,batch_size=10)
    print('Prediction using CB513')
    # pred_cb = model.predict({'main_input': test_hot, 'aux_input': testpssm}, verbose=1,batch_size=1)
    pred_cb = model.predict({'main_input': cb513.test_hot, 'aux_input': cb513.testpssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    testlabel = cb513.testlabel.astype(np.float32)
    pred_cb = pred_cb.astype(np.float32)

    # cat_acc_cb = categorical_accuracy(testlabel, pred_cb)
    # print('CB513 Categorical Accuracy: {}'.format(cat_acc_cb))
    mse_cb = mean_squared_error(testlabel, pred_cb)
    print('CB513 Mean Squared Error: {} '.format(mse_cb))
    mae_cb = mean_absolute_error(testlabel, pred_cb)
    print('CB513 Mean Absolute Error: {} '.format(mae_cb))
    recall_cb = recall(testlabel, pred_cb)
    print('CB513 Recall: {} '.format(recall_cb))
    precision_cb = precision(testlabel, pred_cb)
    print('CB513 Precision: {} '.format(precision_cb))
    f1_score_cb = fbeta_score(testlabel, pred_cb)
    print('CB513 F1 Score: {} '.format(f1_score_cb))

    model_output['CB513 Evaluation Accuracy'] = score[1]
    model_output['CB513 Evaluation Loss'] = score[0]
    # model_output['CB513 Categorical Accuracy'] = cat_acc_cb
    model_output['CB513 Mean Squared Error'] = float(mse_cb.numpy())
    model_output['CB513 Mean Absolute Error'] = float(mae_cb.numpy())
    model_output['CB513 Recall'] = float(recall_cb.numpy())
    model_output['CB513 Precision'] = float(precision_cb.numpy())
    model_output['CB513 F1 Score'] = float(f1_score_cb.numpy())

    get_label_predictions(pred_cb, "CB513")

    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))


'''
Evaluate model using CASP10 test dataset
'''
def evaluate_casp10(model):

    """
    Description:
        Evaluate model using CASP10 test dataset

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    #load test dataset
    casp10 = CASP10()

    print('Evaluating model using CASP10 dataset')
    score = model.evaluate({'main_input': casp10.casp10_data_test_hot, 'aux_input': casp10.casp10_data_pssm},{'main_output': casp10.test_labels},verbose=1,batch_size=1)
    print('CASP10 Evaluation Loss: ', score[0])
    print('CASP10 Evaluation Accuracy: ', score[1])

    print('Prediction using CASP10')
    pred_casp = model.predict({'main_input': casp10.casp10_data_test_hot, 'aux_input': casp10.casp10_data_pssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    test_labels = casp10.test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    mse_casp = mean_squared_error(test_labels, pred_casp)
    print('CASP10 Mean Squared Error: {} '.format(mse_casp))
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    print('CASP10 Mean Absolute Error: {} '.format(mae_casp))
    recall_casp = recall(test_labels, pred_casp)
    print('CASP10 Recall: {} '.format(recall_casp))
    precision_casp = precision(test_labels, pred_casp)
    print('CASP10 Precision: {} '.format(precision_casp))
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    print('CASP10 F1 Score: {} '.format(f1_score_casp))

    model_output['CASP10 Evaluation Accuracy'] = score[1]
    model_output['CASP10 Evaluation Loss'] = score[0]
    model_output['CASP10 Mean Squared Error'] = float(mse_casp.numpy())
    model_output['CASP10 Mean Absolute Error'] = float(mae_casp.numpy())
    model_output['CASP10 Recall'] = float(recall_casp.numpy())
    model_output['CASP10 Precision'] = float(precision_casp.numpy())
    model_output['CASP10 F1 Score'] = float(f1_score_casp.numpy())

    get_label_predictions(pred_casp, "CASP10")

    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))

'''
Evaluate model using CASP11 test dataset
'''
def evaluate_casp11(model):

    """
    Description:
        Evaluate model using CASP10 test dataset

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    #load test dataset
    casp11 = CASP11()

    print('Evaluating model using CASP11 dataset')
    score = model.evaluate({'main_input': casp11.casp11_data_test_hot, 'aux_input': casp11.casp11_data_pssm},{'main_output': casp11.test_labels},verbose=1,batch_size=1)
    print('CASP11 Evaluation Loss: ', score[0])
    print('CASP11 Evaluation Accuracy: ', score[1])

    print('Prediction using CASP11')
    pred_casp = model.predict({'main_input': casp11.casp11_data_test_hot, 'aux_input': casp11.casp11_data_pssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    test_labels = casp11.test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    mse_casp = mean_squared_error(test_labels, pred_casp)
    print('CASP11 Mean Squared Error: {} '.format(mse_casp))
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    print('CASP11 Mean Absolute Error: {} '.format(mae_casp))
    recall_casp = recall(test_labels, pred_casp)
    print('CASP11 Recall: {} '.format(recall_casp))
    precision_casp = precision(test_labels, pred_casp)
    print('CASP11 Precision: {} '.format(precision_casp))
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    print('CASP11 F1 Score: {} '.format(f1_score_casp))

    model_output['CASP11 Evaluation Accuracy'] = score[1]
    model_output['CASP11 Evaluation Loss'] = score[0]
    model_output['CASP11 Mean Squared Error'] = float(mse_casp.numpy())
    model_output['CASP11 Mean Absolute Error'] = float(mae_casp.numpy())
    model_output['CASP11 Recall'] = float(recall_casp.numpy())
    model_output['CASP11 Precision'] = float(precision_casp.numpy())
    model_output['CASP11 F1 Score'] = float(f1_score_casp.numpy())

    get_label_predictions(pred_casp, "CASP11")
    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))

'''
Getting predicted proportion of each secondary structure label
'''
def get_label_predictions(y_pred, calling_test_dataset):

    """
    Description:
        Evaluate model using CASP10 test dataset

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    pred_labels = y_pred[0,0,:]
    print('Model prediction for each secondary structure type using {} dataset:\n'.format(calling_test_dataset))

    model_output[calling_test_dataset + ': '+ 'Alpha Helix (H)'] = pred_labels[5]
    model_output[calling_test_dataset + ': '+ 'Beta Strand (E)'] = pred_labels[2]
    model_output[calling_test_dataset + ': '+ 'Loop (L)'] = pred_labels[0]
    model_output[calling_test_dataset + ': '+ 'Beta Turn (T)'] = pred_labels[7]
    model_output[calling_test_dataset + ': '+ 'Bend (S)'] = pred_labels[6]
    model_output[calling_test_dataset + ': '+ '3-Helix (G)'] = pred_labels[3]
    model_output[calling_test_dataset + ': '+ 'Beta Bridge (B)'] = pred_labels[1]
    model_output[calling_test_dataset + ': '+ 'Pi Helix (I)'] = pred_labels[4]


def categorical_accuracy(y_true, y_pred):
    """
    Description:
        Evaluate model using CASP10 test dataset

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))

def weighted_accuracy(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1),
              K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)

def sparse_categorical_accuracy(y_true, y_pred):
    '''Same as categorical_accuracy, but useful when the predictions are for
    sparse targets.
    '''
    return K.mean(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    '''Calculates the top-k categorical accuracy rate, i.e. success when the
    target class is within the top-k predictions provided.
    '''
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

'''
    Calculates the mean squared error (mse) rate
    between predicted and target values.

'''
def mean_squared_error(y_true, y_pred):

    return K.mean(K.square(y_pred - y_true))

def mean_absolute_error(y_true, y_pred):

    '''Calculates the mean absolute error (mae) rate
    between predicted and target values.
    '''
    return K.mean(K.abs(y_pred - y_true))


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # img_final = tf.image.convert_image_dtype(img_final, tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def poisson(y_true, y_pred):
    '''Calculates the poisson function over prediction and target values.
    '''
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))

def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    # p = tf.cast((precision(y_true, y_pred)),tf.float32)
    # r = tf.cast((recall(y_true, y_pred)),tf.float32)
    # bb = tf.cast((beta ** 2),tf.float32)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


if __name__ == "__main__":

    #initialise input arguments
    parser = argparse.ArgumentParser(description='Evaluating Model')

    parser.add_argument('-model', '--model', required = True,
                    help='Path of model to evaluate')
    parser.add_argument('-test_dataset', '--test_dataset', required = False, default ="all",
                    help='Select what dataset to evaluate model on; default all.')

    #parse arguments
    args = parser.parse_args()
    model = str(args.model)
    test_dataset = str(args.test_dataset)

    #check if model path exists in dir
    if os.path.isfile(model):
        evaluate_model(model, test_dataset)
    else:
        print('Model does not exist...')

#
# def conf_matrix(p, t, num_classes):
#     if p.ndim == 1:
#         p = one_hot(p, num_classes)
#     if t.ndim == 1:
#         t = one_hot(t, num_classes)
#     return np.dot(p.T, t)
