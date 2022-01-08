################################################################################
#######################        Evaluate PSP models       #######################
################################################################################

#import required modules and dependancies
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import *
import tensorflow.keras.backend as K
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import plot_confusion_matrix
try:
    from _globals import *
    from dataset import *
    from plot_model import *
    from utils import *
except:
    from . _globals import *
    from . dataset import *
    from . plot_model import *
    from . utils import *

def evaluate_model(model, test_dataset="all"):
    """
    Description:
        Evaluate model on test datasets.
    Args:
        :model (Keras.model): model to be evaluated.
        :test_dataset (str): test dataset to evaluate on, default - all
        (all 3 available test datasets - CB513, CASP10, CASP11)
    Returns:
        None
    """
    #verify test dataset to be evaluated on, all datasets used by default
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

def evaluate_cullpdb(model, dataset):
    """
    Description:
        Evaluate model using test data from CullPDB unfiltered datasets.
    Args:
        :model (Keras.model): model to be evaluated.
        :dataset (CullPDB instance): instance of cullPDB training dataset to evaluate model on.
    Returns:
        None
    """
    if (dataset.filtered):
        print('Cant evaluate CullPDB using CullPDB proteins if the dataset must is the 6133 or 5926 filtered version.')
        return

    print('Evaluating model using CullPDB {} dataset'.format(dataset.type))
    score = model.evaluate({'main_input': dataset.test_hot, 'aux_input': dataset.test_pssm},
        {'main_output': dataset.test_labels}, verbose=1, batch_size=1)

    #predicting protein labels values for test proteins
    pred_cull = model.predict({'main_input': dataset.test_hot,
        'aux_input': dataset.test_pssm}, verbose=1, batch_size=1)

    #convert label and prediction array to float32
    test_labels = dataset.test_labels.astype(np.float32)
    pred_cull = pred_cull.astype(np.float32)

    #verify observed and predicted labels are the same shape
    if (test_labels.shape!=pred_cull.shape):
        raise ValueError('Observed and predicted values must be the same shape.')

    #get metric results for test dataset
    cat_acc_cull = categorical_accuracy(test_labels, pred_cull)
    mse_cull = mean_squared_error(test_labels, pred_cull)
    rmse_cull = root_mean_square_error(test_labels, pred_cull)
    mae_cull = mean_absolute_error(test_labels, pred_cull)
    recall_cull = recall(test_labels, pred_cull)
    precision_cull = precision(test_labels, pred_cull)
    f1_score_cull = fbeta_score(test_labels, pred_cull)
    fn_cull = FN(test_labels, pred_cull)
    fp_cull = FP(test_labels, pred_cull)
    auc_cull = auc(test_labels, pred_cull)

    #append metric results to global model_output dict
    model_output["CullPDB Evaluation Accuracy"] = score[1]
    model_output["CullPDB Evaluation Loss"] = score[0]
    model_output["CullPDB Categorical Accuracy"] = float(cat_acc_cull.numpy())
    model_output["CullPDB Mean Squared Error"] = float(mse_cull.numpy())
    model_output["CullPDB Root Mean Squared Error"] = float(rmse_cull)
    model_output["CullPDB Mean Absolute Error"] = float(mae_cull.numpy())
    model_output["CullPDB Recall"] = float(recall_cull.numpy())
    model_output["CullPDB Precision"] = float(precision_cull.numpy())
    model_output["CullPDB F1 Score"] = float(f1_score_cull.numpy())
    model_output["CullPDB False Negatives"] = float(fn_cull)
    model_output["CullPDB False Positives"] = float(fp_cull)
    model_output["CullPDB AUC"] = float(auc_cull)

    #get secondary structure label predictions for cullpdb test dataset
    get_label_predictions(pred_cull, "CullPDB")

def evaluate_cb513(model):
    """
    Description:
        Evaluate model using CB513 test dataset.
    Args:
        :model (Keras.model): model to evaluate the CB513 dataset on.
    Returns:
        None
    """
    #load test dataset
    cb513 = CB513()

    print('Evaluating model using CB513 dataset')
    score = model.evaluate({'main_input': cb513.test_hot, 'aux_input': cb513.test_pssm},
        {'main_output': cb513.test_labels}, verbose=1, batch_size=1)

    #predicting protein labels values for test proteins
    pred_cb = model.predict({'main_input': cb513.test_hot, 'aux_input': cb513.test_pssm},
        verbose=1, batch_size=1)

    #convert label and prediction array to float32
    test_labels = cb513.test_labels.astype(np.float32)
    pred_cb = pred_cb.astype(np.float32)

    #verify observed and predicted labels are the same shape
    if (test_labels.shape!=pred_cb.shape):
        raise ValueError('Observed and predicted values must be the same shape.')

    #get metric results for test dataset
    cat_acc_cb = categorical_accuracy(test_labels, pred_cb)
    mse_cb = mean_squared_error(test_labels, pred_cb)
    rmse_cb = root_mean_square_error(test_labels, pred_cb)
    mae_cb = mean_absolute_error(test_labels, pred_cb)
    recall_cb = recall(test_labels, pred_cb)
    precision_cb = precision(test_labels, pred_cb)
    f1_score_cb = fbeta_score(test_labels, pred_cb)
    fn_cb = FN(test_labels, pred_cb)
    fp_cb = FP(test_labels, pred_cb)
    auc_cb = auc(test_labels, pred_cb)

    #append metric results to global model_output dict
    model_output["CB513 Evaluation Accuracy"] = score[1]
    model_output["CB513 Evaluation Loss"] = score[0]
    model_output["CB513 Categorical Accuracy"] = float(cat_acc_cb.numpy())
    model_output["CB513 Mean Squared Error"] = float(mse_cb.numpy())
    model_output["CB513 Root Mean Squared Error"] = float(rmse_cb.numpy())
    model_output["CB513 Mean Absolute Error"] = float(mae_cb.numpy())
    model_output["CB513 Recall"] = float(recall_cb.numpy())
    model_output["CB513 Precision"] = float(precision_cb.numpy())
    model_output["CB513 F1 Score"] = float(f1_score_cb.numpy())
    model_output["CB513 False Negatives"] = float(fn_cb)
    model_output["CB513 False Positives"] = float(fp_cb)
    model_output["CB513 AUC"] = float(auc_cb)

    #get secondary structure label predictions for cb513 test dataset
    get_label_predictions(pred_cb, "CB513")

def evaluate_casp10(model):
    """
    Description:
        Evaluate model using CASP10 test dataset.
    Args:
        :model (Keras.model): model to evaluate the CASP10 dataset on.
    Returns:
        None
    """
    #load test dataset
    casp10 = CASP10()

    print('Evaluating model using CASP10 dataset')
    score = model.evaluate({'main_input': casp10.test_hot,
        'aux_input': casp10.test_pssm},{'main_output': casp10.test_labels},verbose=1, batch_size=1)

    #predicting protein labels values for test proteins
    pred_casp = model.predict({'main_input': casp10.test_hot,
        'aux_input': casp10.test_pssm}, verbose=1, batch_size=1)

    #convert label and prediction array to float32
    test_labels = casp10.test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    #verify observed and predicted labels are the same shape
    if (test_labels.shape!=pred_casp.shape):
        raise ValueError('Observed and predicted values must be the same shape.')

    #get metric results for test dataset
    cat_acc_casp = categorical_accuracy(test_labels, pred_casp)
    mse_casp = mean_squared_error(test_labels, pred_casp)
    rmse_casp = root_mean_square_error(test_labels, pred_casp)
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    recall_casp = recall(test_labels, pred_casp)
    precision_casp = precision(test_labels, pred_casp)
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    fn_casp = FN(test_labels, pred_casp)
    fp_casp = FP(test_labels, pred_casp)
    auc_casp = auc(test_labels, pred_casp)

    #append metric results to global model_output dict
    model_output["CASP10 Evaluation Accuracy"] = score[1]
    model_output["CASP10 Evaluation Loss"] = score[0]
    model_output["CASP10 Cateogorical Accuracy"] = float(cat_acc_casp.numpy())
    model_output["CASP10 Mean Squared Error"] = float(mse_casp.numpy())
    model_output["CASP10 Root Mean Squared Error"] = float(rmse_casp.numpy())
    model_output["CASP10 Mean Absolute Error"] = float(mae_casp.numpy())
    model_output["CASP10 Recall"] = float(recall_casp.numpy())
    model_output["CASP10 Precision "] = float(precision_casp.numpy())
    model_output["CASP10 F1 Score"] = float(f1_score_casp.numpy())
    model_output["CASP10 False Negatives"] = float(fn_casp)
    model_output["CASP10 False Positives"] = float(fp_casp)
    model_output["CASP10 AUC"] = float(auc_casp)

    #get secondary structure label predictions for CASP10 test dataset
    get_label_predictions(pred_casp, "CASP10")

def evaluate_casp11(model):
    """
    Description:
        Evaluate model using CASP11 test dataset.
    Args:
        :model (Keras.model): model to evaluate the CASP11 dataset on.
    Returns:
        None
    """
    #load test dataset
    casp11 = CASP11()

    print('Evaluating model using CASP11 dataset')
    score = model.evaluate({'main_input': casp11.test_hot,
        'aux_input': casp11.test_pssm},{'main_output': casp11.test_labels},verbose=1, batch_size=1)

    #predicting protein labels values for test proteins
    pred_casp = model.predict({'main_input': casp11.test_hot,
        'aux_input': casp11.test_pssm}, verbose=1, batch_size=1)

    #convert label and prediction array to float32
    test_labels = casp11.test_labels.astype(np.float32)
    pred_casp = pred_casp.astype(np.float32)

    #verify observed and predicted labels are the same shape
    if (test_labels.shape!=pred_casp.shape):
        raise ValueError('Observed and predicted values must be the same shape.')

    #get metric results for test dataset
    cat_acc_casp = categorical_accuracy(test_labels, pred_casp)
    mse_casp = mean_squared_error(test_labels, pred_casp)
    rmse_casp = root_mean_square_error(test_labels, pred_casp)
    mae_casp = mean_absolute_error(test_labels, pred_casp)
    recall_casp = recall(test_labels, pred_casp)
    precision_casp = precision(test_labels, pred_casp)
    f1_score_casp = fbeta_score(test_labels, pred_casp)
    fn_casp = FN(test_labels, pred_casp)
    fp_casp = FP(test_labels, pred_casp)
    auc_casp = auc(test_labels, pred_casp)

    #append metric results to global model_output dict
    model_output["CASP11 Evaluation Accuracy"] = score[1]
    model_output["CASP11 Evaluation Loss"] = score[0]
    model_output["CASP11 Cateogorical Accuracy"] = float(cat_acc_casp.numpy())
    model_output["CASP11 Mean Squared Error"] = float(mse_casp.numpy())
    model_output["CASP11 Root Mean Squared Error"] = float(rmse_casp.numpy())
    model_output["CASP11 Mean Absolute Error"] = float(mae_casp.numpy())
    model_output["CASP11 Recall"] = float(recall_casp.numpy())
    model_output["CASP11 Precision "] = float(precision_casp.numpy())
    model_output["CASP11 F1 Score"] = float(f1_score_casp.numpy())
    model_output["CASP11 False Negatives"] = float(fn_casp)
    model_output["CASP11 False Positives"] = float(fp_casp)
    model_output["CASP11 AUC"] = float(auc_casp)

    #get secondary structure label predictions for CASP11 test dataset
    get_label_predictions(pred_casp, "CASP11")

def get_label_predictions(y_pred, calling_test_dataset):
    """
    Description:
        Get predicted class predictions for each secondary structure label.
    Args:
        :y_pred (np.ndarray): predicted class labels.
        :calling_test_dataset (str): name of test dataset calling function (CullPDB, CB513, CASP10/11).
    Returns:
        None
    """
    #predicted labels are in 3rd dimension of y_pred array
    pred_labels = y_pred[0,0,:]

    #output proportion of each predicted protein label to model_output dict
    model_output[calling_test_dataset + ': '+ 'Alpha Helix (H)'] = pred_labels[5]
    model_output[calling_test_dataset + ': '+ 'Beta Strand (E)'] = pred_labels[2]
    model_output[calling_test_dataset + ': '+ 'Loop (L)'] = pred_labels[0]
    model_output[calling_test_dataset + ': '+ 'Beta Turn (T)'] = pred_labels[7]
    model_output[calling_test_dataset + ': '+ 'Bend (S)'] = pred_labels[6]
    model_output[calling_test_dataset + ': '+ '3-Helix (G)'] = pred_labels[3]
    model_output[calling_test_dataset + ': '+ 'Beta Bridge (B)'] = pred_labels[1]
    model_output[calling_test_dataset + ': '+ 'Pi Helix (I)'] = pred_labels[4]

    #plot proportion of each protein label predicted
    plot_protein_labels(pred_labels, calling_test_dataset)

def categorical_accuracy(y_true, y_pred):
    """
    Description:
        Calcualte mean accuracy rate across all predictions for multiclass classification.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        categorical_accuracy (float).
    """
    return (K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1))))

def weighted_accuracy(y_true, y_pred):
    """
    Description:
        Calculate weighted accuracy - taking the average, over all the classes, of the fraction of correct predictions over
        the total number of samples.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        weighted_accuracy (float).
    """
    return (K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true))

def sparse_categorical_accuracy(y_true, y_pred):
    """
    Description:
        Same as categorical_accuracy, but useful when the predictions are for sparse targets.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        sparse_categorical_accuracy (float).
    """
    return K.mean(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))

def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """
    Description:
        Calculates the top-k categorical accuracy rate - success when the target class is within
        the top-k predictions provided.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
        :k (int): upper bound for number of predictions that targets are predicted to be in.
    Returns:
        top_k_categorical_accuracy (float).
    """
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

def mean_squared_error(y_true, y_pred):
    """
    Description:
        Calculates the mean squared error (mse) rate between predicted and target values.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        mean_squared_error (float).
    """
    return K.mean(K.square(y_pred - y_true))

def root_mean_square_error(y_true, y_pred):
    """
    Description:
        Calculates the root mean squared error (rmse) rate between predicted and target values.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        root_mean_squared_error (float).
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_absolute_error(y_true, y_pred):
    """
    Description:
        Calculates the mean absolute error (mae) rate between predicted and target values.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        mean_absolute_error (float).
    """
    return K.mean(K.abs(y_pred - y_true))

def precision(y_true, y_pred):
    """
    Description:
        Calculates the precision - positive predictive value, fraction of relevant instances
        among the retrieved instances; true positives / true positives + false positives.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        precision (float).
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def recall(y_true, y_pred):
    """
    Description:
        Calculates the recall - sensitivity,  is the fraction of the total amount of
        relevant instances that were actually retrieved.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        recall (float)
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def FN(y_true, y_pred):
    """
    Description:
        Calculate the number of false negatives during prediction.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        false negatives (float) - the total number of false negatives during evaluation.
    """
    # FN = np.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    fn = int(possible_positives - true_positives)

    return fn

def FP(y_true, y_pred):
    """
    Description:
        Calculate the number of false positives during prediction.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        false positives (float) - the total number of false positives during evaluation.
    """
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    fp = int(predicted_positives - true_positives)

    return fp

def auc(y_true, y_pred):
    """
    Description:
        Calculate the area under the curve (AUC).
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        area under curve auc (float) - the total number of false positives during evaluation.
    """
    auc_ = tf.keras.metrics.AUC(num_thresholds=3)
    auc_.update_state(y_true, y_pred)
    auc_result = auc_.result().numpy()

    # auc = tf.keras.metrics.AUC(y_true, y_pred)#[1]
    return auc_result

def poisson(y_true, y_pred):
    """
    Description:
        Calculates the poisson function over prediction and target values.
    Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
    Returns:
        poisson (float).
    """
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))

def fbeta_score(y_true, y_pred, beta=1):
    """
    Description:
        Calculates the F score, the weighted harmonic mean of precision and recall.
        This is useful for multi-label classification, where input samples can be
        classified as sets of labels.
     Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class labels.
        :beta (int): The F-beta score (ranging from 0.0 to 1.0) is the weighted mean of the
                proportion of correct class assignments vs. the proportion of
                incorrect class assignments. With beta = 1, this is equivalent to a F-measure.
    Returns:
        fbeta_score (float).
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score

def confusion_matrix_(y_true, y_pred):
    """
    Description:
        Build confusion matrix of predicted and observed secondary structure labels.
     Args:
        :y_true (np.ndarray): ground truth class labels.
        :y_pred (np.ndarray): predicted class label.
    Returns:
        matrix(float): matrix of predicted vs observed.
    """
    return confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

if __name__ == "__main__":

    #initialise input arguments
    parser = argparse.ArgumentParser(description='Evaluating Model')

    parser.add_argument('-model_path', '--model_path', required = True,
                    help='Path of model to evaluate')
    parser.add_argument('-test_dataset', '--test_dataset', required = False, default ="all",
                    help='Select what dataset to evaluate model on; default all.')

    #parse arguments
    args = parser.parse_args()
    model_path = str(args.model_path)
    test_dataset = str(args.test_dataset)

    #check if model path exists in dir
    if os.path.isfile(model_path):
        evaluate_model(model_path, test_dataset)
    else:
        print('Model does not exist...')
        sys.exit(0)
