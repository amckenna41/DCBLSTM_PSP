#######################################
### Evaluate CDULSTM/CDBLSTM models ###
#######################################

import numpy as np
from keras import backend as K
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from globals import *

def evaluate_model(model, test_dataset="all"):

    """
    Description:
        Evaluate model using test datasets

    Args:

    Returns:
        None
    """

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

def evaluate_cb513(model):

    """
    Description:
        Evaluate model using CB513 test dataset

    Args:
        model (keras/TF model): model to evaluate the CB513 dataset on

    Returns:
        None
    """

    #load test dataset
    test_hot, testpssm, testlabel = load_cb513()

    print('Evaluating model using CB513 dataset')
    score = model.evaluate({'main_input': test_hot, 'aux_input': testpssm},{'main_output': testlabel},verbose=1,batch_size=1)
    print('Evaluation Loss : ', score[0])
    print('Evaluation Accuracy : ', score[1])

    # pred_casp = model.predict([testpssm, test_hot], verbose=1,batch_size=10)
    print('Prediction using CB513')
    pred_cb = model.predict({'main_input': test_hot, 'aux_input': testpssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    testlabel = testlabel.astype(np.float32)
    pred_cb = pred_cb.astype(np.float32)

    cat_acc_casp = categorical_accuracy(testlabel, pred_cb)
    print('Categorical Accuracy: {}'.format(cat_acc_casp))
    mse_cb = mean_squared_error(testlabel, pred_cb)
    print('Mean Sqared Error: {} '.format(mse_cb))
    mae_cb = mean_absolute_error(testlabel, pred_cb)
    print('Mean Absolute Error: {} '.format(mae_cb))
    recall_cb = recall(testlabel, pred_cb)
    print('Recall: {} '.format(recall_cb))
    precision_cb = precision(testlabel, pred_cb)
    print('Precision: {} '.format(precision_cb))
    f1_score_cb = fbeta_score(testlabel, pred_cb)
    print('F1 Score: {} '.format(f1_score_cb))

    get_label_predictions(pred_cb)

    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))


def evaluate_casp10(model):

    """
    Description:
        Evaluate model using CASP10 test dataset

    Args:
        model (keras/TF model): model to evaluate the CASP10 dataset on

    Returns:
        None
    """

    #load test dataset
    casp10_data_test_hot, casp10_data_pssm, test_labels = load_casp10()

    print('Evaluating model using CASP10 dataset')
    score = model.evaluate({'main_input': casp10_data_test_hot, 'aux_input': casp10_data_pssm},{'main_output': test_labels},verbose=1,batch_size=1)
    print('Evaluation Loss : ', score[0])
    print('Evaluation Accuracy : ', score[1])

    print('Prediction using CASP10')
    pred_casp = model.predict({'main_input': casp10_data_test_hot, 'aux_input': casp10_data_pssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    test_labels = test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    mse_casp = mean_squared_error(test_labels, pred_casp)
    print('Mean Squared Error - {} '.format(mse_casp))
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    print('Mean Absolute Error - {} '.format(mae_casp))
    recall_casp = recall(test_labels, pred_casp)
    print('Recall - {} '.format(recall_casp))
    precision_casp = precision(test_labels, pred_casp)
    print('Precision - {} '.format(precision_casp))
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    print('F1 Score - , {} '.format(f1_score_casp))

    get_label_predictions(pred_casp)

    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))

def evaluate_casp11(model):

    """
    Description:
        Evaluate model using CASP11 test dataset

    Args:
        model (keras/TF model): model to evaluate the CASP11 dataset on

    Returns:
        None
    """

    #load test dataset
    casp11_data_test_hot, casp11_data_test_pssm, test_labels = load_casp11()

    print('Evaluating model using CASP11 dataset')
    score = model.evaluate({'main_input': casp11_data_test_hot, 'aux_input': casp11_data_test_pssm},{'main_output': test_labels},verbose=1,batch_size=1)
    print('Evaluation Loss : ', score[0])
    print('Evaluation Accuracy : ', score[1])

    print('Prediction using CASP11')
    pred_casp = model.predict({'main_input': casp11_data_test_hot, 'aux_input': casp11_data_test_pssm}, verbose=1,batch_size=1)

    #convert label and prediction array to float32
    test_labels = test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    mse_casp = mean_squared_error(test_labels, pred_casp)
    print('Mean Squared Error - {} '.format(mse_casp))
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    print('Mean Absolute Error - {} '.format(mae_casp))
    recall_casp = recall(test_labels, pred_casp)
    print('Recall - {} '.format(recall_casp))
    precision_casp = precision(test_labels, pred_casp)
    print('Precision - {} '.format(precision_casp))
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    print('F1 Score - , {} '.format(f1_score_casp))

    get_label_predictions(pred_casp)
    # print(classification_report(testlabel, pred_casp, target_names=class_names))
    # print(confusion_matrix(testlabel, pred_casp))

'''
Getting predicted proportion of each secondary structure label
'''
def get_label_predictions(y_pred):

    pred_labels = y_pred[0,0,:]
    print('Model prediction for each secondary structure type:\n')
    print('Alpha Helix (H): ', pred_labels[5])
    print('Beta Strand (E): ', pred_labels[2])
    print('Loop (L): ', pred_labels[0])
    print('Beta Turn (T): ', pred_labels[7])
    print('Bend (S): ', pred_labels[6])
    print('3-Helix (G): ',pred_labels[3])
    print('Beta Bridge (B): ', pred_labels[1])
    print('Pi Helix (I): ', pred_labels[4])

def categorical_accuracy(y_true, y_pred):

    """
    Description:
        Calcualte mean accuracy rate across all predictions for multiclass
        classification.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))

def weighted_accuracy(y_true, y_pred):

    """
    Description:
        Calcualte mean accuracy rate across all predictions for multiclass
        classification.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    return K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)

def sparse_categorical_accuracy(y_true, y_pred):

    """
    Description:
        Same as categorical_accuracy, but useful when the predictions are for
        sparse targets.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    return K.mean(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))


def top_k_categorical_accuracy(y_true, y_pred, k=5):

    """
    Description:
        Calculates the top-k categorical accuracy rate, i.e. success when the
        target class is within the top-k predictions provided.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))


def mean_squared_error(y_true, y_pred):

    """
    Description:
        Calculates the mean squared error (mse) rate
        between predicted and target values.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """
    return K.mean(K.square(y_pred - y_true))

def mean_absolute_error(y_true, y_pred):

    """
    Description:
        Calculates the mean absolute error (mae) rate
        between predicted and target values.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """
    return K.mean(K.abs(y_pred - y_true))


def precision(y_true, y_pred):

    """
    Description:
        Calculates the precision, a metric for multi-label classification of
        how many selected items are relevant.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # img_final = tf.image.convert_image_dtype(img_final, tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):

    """
    Description:
        Calculates the precision, a metric for multi-label classification of
        how many selected items are relevant.

    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def poisson(y_true, y_pred):

    """
    Description:
        Calculates the poisson function over prediction and target values.
    Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """

    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))

def fbeta_score(y_true, y_pred, beta=1):

    """
    Description:
        Calculates the F score, the weighted harmonic mean of precision and recall.
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
     Args:
        y_true (np.ndarray):
        y_pred (np.ndarray):

    Returns:
        categorical_accuracy (float)
    """
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
        return