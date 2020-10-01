# import numpy as np
# from keras import optimizers, callbacks
# from timeit import default_timer as timer
# from dataset import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513
# import model
#
# dataset = get_dataset()
#
# D_train, D_test, D_val = split_with_shuffle(dataset, 100)
#
# X_train, Y_train = get_data_labels(D_train)
# X_test, Y_test = get_data_labels(D_test)
# X_val, Y_val = get_data_labels(D_val)
#
# net = model.CNN_model()
#
# #load Weights
# net.load_weights("Whole_CullPDB-best.hdf5")
#
# predictions = net.predict(X_test)
#
# print("\n\nQ8 accuracy: " + str(model.Q8_accuracy(Y_test, predictions)) + "\n\n")
#
# CB513_X, CB513_Y = get_cb513()
#
# predictions = net.predict(CB513_X)
#
# print("\n\nQ8 accuracy on CB513: " + str(model.Q8_accuracy(CB513_Y, predictions)) + "\n\n")
# © 2020 GitHub, Inc.
# 
# import numpy as np
# from . import backend as K
# from .utils.generic_utils import get_from_module
#
#
# def binary_accuracy(y_true, y_pred):
#     '''Calculates the mean accuracy rate across all predictions for binary
#     classification problems.
#     '''
#     return K.mean(K.equal(y_true, K.round(y_pred)))
