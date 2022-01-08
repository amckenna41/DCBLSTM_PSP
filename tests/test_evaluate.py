################################################################################
##########                Tests for evaluation module                 ##########
################################################################################

import os
import requests
import numpy as np
import json
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from psp.dataset import *
from psp.evaluate import *
import psp.models.dummy_model as dummy_model

class EvaluationTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Initialise dataset class objects. """

        #open JSON config file in config folder
        with open(os.path.join("config","dummy.json")) as f:
            self.dummy_model_params = json.load(f)

        self.cullpdb = CullPDB(type='6133', filtered=0)
        # self.cullpdb_6133 = CullPDB(type='6133', filtered=0)
        # self.cullpdb_5926 = CullPDB(type='5926', filtered=0)
        self.cb513 = CB513()
        self.casp10 = CASP10()
        self.casp11 = CASP11()

        #build model
        self.dummy_model_ = dummy_model.build_model(self.dummy_model_params["model_parameters"][0])

        #predicting protein labels values for test proteins in CullPDB dataset
        self.score = self.dummy_model_.evaluate({'main_input': self.cullpdb.test_hot, 'aux_input': self.cullpdb.test_pssm},
            {'main_output': self.cullpdb.test_labels}, verbose=1, batch_size=1)
        self.pred_cullpdb = self.dummy_model_.predict({'main_input': self.cullpdb.test_hot, 'aux_input': self.cullpdb.test_pssm}, 
            verbose=1, batch_size=1)
        self.pred_cullpdb = self.pred_cullpdb.astype(np.float32)

        #predicting protein labels values for test proteins in CB513 dataset
        self.pred_cb513 = self.dummy_model_.predict({'main_input': self.cb513.test_hot,
            'aux_input': self.cb513.test_pssm}, verbose=1, batch_size=1)
        self.pred_cb513 = self.pred_cb513.astype(np.float32)

        #predicting protein labels values for test proteins in CASP10 dataset
        self.pred_casp10 = self.dummy_model_.predict({'main_input': self.casp10.test_hot,
            'aux_input': self.casp10.test_pssm}, verbose=1, batch_size=1)
        self.pred_casp10 = self.pred_casp10.astype(np.float32)

        #predicting protein labels values for test proteins in CASP11 dataset
        self.pred_casp11 = self.dummy_model_.predict({'main_input': self.casp11.test_hot,
            'aux_input': self.casp11.test_pssm}, verbose=1, batch_size=1)
        self.pred_casp11 = self.pred_casp11.astype(np.float32)

        #ground-truth labels
        self.cullpdb_test_labels = self.cullpdb.test_labels.astype(np.float32)
        self.cb513_test_labels = self.cb513.test_labels
        self.casp10_test_labels = self.casp10.test_labels
        self.casp11_test_labels = self.casp11.test_labels

    def test_prediction(self):
        """ Testing prediction array shapes and data types. """
#1.)
        self.assertEqual(self.pred_cullpdb.shape, (272, 700, 8))
        self.assertTrue(type(self.pred_cullpdb).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(self.pred_cullpdb.dtype, 'float32')
#2.)
        self.assertEqual(self.pred_cb513.shape, (514, 700, 8))
        self.assertTrue(type(self.pred_cb513).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(self.pred_cb513.dtype, 'float32')
#3.)    
        self.assertEqual(self.pred_casp10.shape, (123, 700, 8))
        self.assertTrue(type(self.pred_casp10).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(self.pred_casp10.dtype, 'float32')
#4.)
        self.assertEqual(self.pred_casp11.shape, (105, 700, 8))
        self.assertTrue(type(self.pred_casp11).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(self.pred_casp11.dtype, 'float32')

    def test_categorical_accuracy(self):
        """ Testing cateogorical accuracy evaluation function. """
#1.)
        cat_acc_cullpdb = categorical_accuracy(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(cat_acc_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(cat_acc_cullpdb >= 0 and cat_acc_cullpdb <=1)
#2.)
        cat_acc_cb513 = categorical_accuracy(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(cat_acc_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(cat_acc_cb513 >= 0 and cat_acc_cb513 <=1)
#3.)
        cat_acc_casp10 = categorical_accuracy(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(cat_acc_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(cat_acc_casp10 >= 0 and cat_acc_casp10 <=1)
#4.)
        cat_acc_casp11 = categorical_accuracy(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(cat_acc_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(cat_acc_casp11 >= 0 and cat_acc_casp11 <=1)

    @unittest.expectedFailure
    def test_weighted_accuracy(self):
        """ Testing weighted accuracy evaluation function. """
#1.)    
        weight_acc_cullpdb = weighted_accuracy(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(weight_acc_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(weight_acc_cullpdb >= 0 and weight_acc_cullpdb <=1)
#2.)
        weight_acc_cb513 = weighted_accuracy(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(weight_acc_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(weight_acc_cb513 >= 0 and weight_acc_cb513 <=1)
#3.)
        weight_acc_casp10 = weighted_accuracy(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(weight_acc_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(weight_acc_casp10 >= 0 and weight_acc_casp10 <=1)
#4.)
        cat_acc_casp11 = weighted_accuracy(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(weight_acc_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(weight_acc_casp11 >= 0 and weight_acc_casp11 <=1)

    @unittest.expectedFailure
    def test_sparse_categorical_accuracy(self):
        """ Testing sparse categorical accuracy evaluation function. """
#1.)
        sparse_cat_acc_cullpdb = sparse_categorical_accuracy(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(sparse_cat_acc_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(sparse_cat_acc_cullpdb >= 0 and sparse_cat_acc_cullpdb <=1)
#2.)
        sparse_cat_acc_cb513 = sparse_categorical_accuracy(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(sparse_cat_acc_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(sparse_cat_acc_cb513 >= 0 and sparse_cat_acc_cb513 <=1)
#3.)
        sparse_cat_acc_casp10 = sparse_categorical_accuracy(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(sparse_cat_acc_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(sparse_cat_acc_casp10 >= 0 and sparse_cat_acc_casp10 <=1)
#4.)
        sparse_cat_acc_casp11 = sparse_categorical_accuracy(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(sparse_cat_acc_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(sparse_cat_acc_casp11 >= 0 and sparse_cat_acc_casp11 <=1)

    @unittest.expectedFailure
    def test_top_k_categorical_accuracy(self):
        """ Testing top-k categorical accuracy evaluation function. """
#1.)
        topk_cat_acc_cullpdb = top_k_categorical_accuracy(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(topk_cat_acc_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(topk_cat_acc_cullpdb >= 0 and topk_cat_acc_cullpdb <=1)
#2.)
        topk_cat_acc_cb513 = top_k_categorical_accuracy(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(topk_cat_acc_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(topk_cat_acc_cb513 >= 0 and topk_cat_acc_cb513 <=1)
#3.)
        topk_cat_acc_casp10 = top_k_categorical_accuracy(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(topk_cat_acc_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(topk_cat_acc_casp10 >= 0 and topk_cat_acc_casp10 <=1)
#4.)
        topk_cat_acc_casp11 = top_k_categorical_accuracy(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(topk_cat_acc_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(topk_cat_acc_casp11 >= 0 and topk_cat_acc_casp11 <=1)

    def test_mean_square_error(self):
        """ Testing mean square error evaluation function. """
#1.)
        mse_cullpdb = mean_squared_error(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(mse_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mse_cullpdb >= 0 and mse_cullpdb <=1)
#2.)
        mse_cb513 = mean_squared_error(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(mse_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mse_cb513 >= 0 and mse_cb513 <=1)
#3.)
        mse_casp10 = mean_squared_error(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(mse_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mse_casp10 >= 0 and mse_casp10 <=1)
#4.)
        mse_casp11 = mean_squared_error(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(mse_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mse_casp11 >= 0 and mse_casp11 <=1)

    def test_root_mean_square_error(self):
        """ Testing root mean square error evaluation function. """
#1.)
        rmse_cullpdb = root_mean_square_error(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(rmse_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(rmse_cullpdb >= 0 and rmse_cullpdb <=1)
#2.)
        rmse_cb513 = root_mean_square_error(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(rmse_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(rmse_cb513 >= 0 and rmse_cb513 <=1)
#3.)
        rmse_casp10 = root_mean_square_error(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(rmse_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(rmse_casp10 >= 0 and rmse_casp10 <=1)
#4.)
        rmse_casp11 = root_mean_square_error(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(rmse_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(rmse_casp11 >= 0 and rmse_casp11 <=1)

    def test_mean_absolute_error(self):
        """ Testing mean absolute error evaluation function. """
#1.)
        mae_cullpdb = mean_absolute_error(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(mae_cullpdb.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mae_cullpdb >= 0 and mae_cullpdb <=1)
#2.)
        mae_cb513 = mean_absolute_error(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(mae_cb513.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mae_cb513 >= 0 and mae_cb513 <=1)
#3.)
        mae_casp10 = mean_absolute_error(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(mae_casp10.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mae_casp10 >= 0 and mae_casp10 <=1)
#4.)
        mae_casp11 = mean_absolute_error(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(mae_casp11.numpy()).__module__ == (np.__name__),'Data type must be numpy array')
        self.assertTrue(mae_casp11 >= 0 and mae_casp11 <=1)

    @unittest.expectedFailure
    def test_precision(self):
        """ Testing precision evaluation function. """
#1.)
        precision_cullpdb = precision(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(type(precision_cullpdb) == float,'Data type must be float')
        self.assertTrue(precision_cullpdb >= 0 and precision_cullpdb <=1)
#2.)
        precision_cb513 = precision(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(type(precision_cb513) == float,'Data type must be float')
        self.assertTrue(precision_cb513 >= 0 and precision_cb513 <=1)
#3.)
        precision_casp10 = precision(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(type(precision_casp10) == float,'Data type must be float')
        self.assertTrue(precision_casp10 >= 0 and precision_casp10 <=1)
#4.)
        precision_casp11 = precision(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(type(precision_casp11) == float,'Data type must be float')
        self.assertTrue(precision_casp11 >= 0 and precision_casp11 <=1)

    @unittest.expectedFailure
    def test_recall(self):
        """ Testing recall evaluation function. """
#1.)
        recall_cullpdb = recall(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(recall_cullpdb.numpy(),np.float64),'Data type must be numpy array')
        self.assertTrue(recall_cullpdb.numpy() >= 0 and recall_cullpdb.numpy() <=1)
#2.)
        recall_cb513 = recall(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(recall_cb513.numpy(),np.float64),'Data type must be numpy array')
        self.assertTrue(recall_cb513.numpy() >= 0 and recall_cb513.numpy() <=1)
#3.)
        recall_casp10 = recall(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(recall_casp10.numpy(),np.float64),'Data type must be numpy array')
        self.assertTrue(recall_casp10.numpy() >= 0 and recall_casp10.numpy() <=1)
#4.)
        recall_casp11 = recall(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(recall_casp11.numpy(),np.float64),'Data type must be numpy array')
        self.assertTrue(recall_casp11.numpy() >= 0 and recall_casp11.numpy() <=1)

    @unittest.expectedFailure
    def test_fn(self):
        """ Testing false negatives evaluation function. """
#1.)
        fn_cullpdb = FN(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(fn_cullpdb, int),'Data type must be int')
#2.)
        fn_cb513 = FN(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(fn_cb513, int),'Data type must be int')
#3.)
        fn_casp10 = FN(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(fn_casp10, int),'Data type must be int')
#4.)
        fn_casp11 = FN(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(fn_casp11, int),'Data type must be int')

    @unittest.expectedFailure
    def test_fp(self):
        """ Testing false positives evaluation function. """
#1.)
        fp_cullpdb = FP(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(fp_cullpdb, int),'Data type must be int')
#2.)
        fp_cb513 = FP(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(fp_cb513, int),'Data type must be int')
#3.)
        fp_casp10 = FP(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(fp_casp10, int),'Data type must be int')
#4.)
        fp_casp11 = FP(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(fp_casp11, int),'Data type must be int')

    def test_auc(self):
        """ Testing AUC evaluation function. """
#1.)
        auc_cullpdb = auc(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(auc_cullpdb,np.float32),'Data type must be numpy array')
        self.assertTrue(auc_cullpdb >= 0 and auc_cullpdb <=1)
#2.)
        auc_cb513 = auc(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(auc_cb513,np.float32),'Data type must be numpy array')
        self.assertTrue(auc_cb513 >= 0 and auc_cb513 <=1)
#3.)
        auc_casp10 = auc(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(auc_casp10,np.float32),'Data type must be numpy array')
        self.assertTrue(auc_casp10 >= 0 and auc_casp10 <=1)
#4.)
        auc_casp11 = auc(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(auc_casp11,np.float32),'Data type must be numpy array')
        self.assertTrue(auc_casp11 >= 0 and auc_casp11 <=1)

    @unittest.expectedFailure
    def test_poission(self):
        """ Testing Poission evaluation function. """
#1.)
        poission_cullpdb = auc(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(poission_cullpdb,np.float32),'Data type must be numpy array')
        self.assertTrue(poission_cullpdb >= 0 and poission_cullpdb <=1)
#2.)
        poission_cb513 = auc(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(poission_cb513,np.float32),'Data type must be numpy array')
        self.assertTrue(poission_cb513 >= 0 and poission_cb513 <=1)
#3.)
        poission_casp10 = auc(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(poission_casp10,np.float32),'Data type must be numpy array')
        self.assertTrue(poission_casp10 >= 0 and poission_casp10 <=1)
#4.)
        poission_casp11 = auc(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(poission_casp11,np.float32),'Data type must be numpy array')
        self.assertTrue(poission_casp11.numpy() >= 0 and poission_casp11.numpy() <=1)

    def test_fbeta_score(self):
        """ Testing fbeta score evaluation function. """
#1.)
        fbeta_cullpdb = auc(self.cullpdb.test_labels, self.pred_cullpdb)
        self.assertTrue(isinstance(fbeta_cullpdb,np.float32),'Data type must be numpy array')
        self.assertTrue(fbeta_cullpdb >= 0 and fbeta_cullpdb <=1)
#2.)
        fbeta_cb513 = auc(self.cb513.test_labels, self.pred_cb513)
        self.assertTrue(isinstance(fbeta_cb513,np.float32),'Data type must be numpy array')
        self.assertTrue(fbeta_cb513 >= 0 and fbeta_cb513 <=1)
#3.)
        fbeta_casp10 = auc(self.casp10.test_labels, self.pred_casp10)
        self.assertTrue(isinstance(fbeta_casp10,np.float32),'Data type must be numpy array')
        self.assertTrue(fbeta_casp10 >= 0 and fbeta_casp10 <=1)
#4.)
        fbeta_casp11 = auc(self.casp11.test_labels, self.pred_casp11)
        self.assertTrue(isinstance(fbeta_casp11,np.float32),'Data type must be numpy array')
        self.assertTrue(fbeta_casp11 >= 0 and fbeta_casp11 <=1)

    @classmethod
    def tearDownClass(self):
        """ Delete all datasets stored in memory and any temp dirs. """

        del self.cullpdb 
        # del self.cullpdb_6133
        # del self.cullpdb_5926
        del self.cb513 
        del self.casp10 
        del self.casp11 
        del self.dummy_model_


if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)
