################################################################################
##########                Tests for evaluation module                 ##########
################################################################################

import os
import requests
import numpy as np
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from psp.load_dataset import *
from psp.evaluate import *

class EvaluationTests(unittest.TestCase):

    def setUp(self):
        """ Initialise dataset class objects. """

        #create temp dir where test data is stored
        os.mkdir("temp_data")

        #build dummy model
        #create dummy weights
        #assert y_true.shape==y_pred.shape

    #
    #     self.my_dense = tf.keras.layers.Dense(2)
    #     self.my_dense.build((2, 2))
    #
    # def testDenseLayerOutput(self):
    #     self.my_dense.set_weights([
    #         np.array([[1, 0],
    #                   [2, 3]]),
    #         np.array([0.5, 0])
    #     ])
    #     input_x = np.array([[1, 2],
    #                        [2, 3]])
    #     output = self.my_dense(input_x)
    #     expected_output = np.array([[5.5, 6.],
    #                                 [8.5, 9]])
    #
    #     self.assertAllEqual(expected_output, output)
    #

    def test_categorical_accuracy(self):
        pass

    def test_weighted_accuracy(self):
        pass

    def test_sparse_categorical_accuracy(self):
        pass

    def test_top_k_categorical_accuracy(self):
        pass

    def test_mean_square_error(self):
        pass

    def test_root_mean_square_error(self):
        pass

    def test_mean_absolute_error(self):
        pass

    def test_precision(self):
        pass

    def test_recall(self):
        pass

    def test_fn(self):
        pass

    def test_fp(self):
        pass

    def test_auc(self):
        pass

    def test_poission(self):
        pass

    def test_fbeta_score(self):
        pass

    def tearDown(self):
        """ Delete all datasets stored in memory and any temp dirs. """

        os.rmdir("temp_data")
    
    def test_evaluation(self):
        print('Evaluation tests')
