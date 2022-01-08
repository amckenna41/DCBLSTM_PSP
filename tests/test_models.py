
################################################################################
############                    Tests for Models                    ############
################################################################################

import os
import sys
sys.path.append('../')
# sys.path.append('models')
# sys.path.append(os.path.join('models','auxiliary_models'))
import json
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
from config import *
from psp.utils import *
import psp.models.dummy_model as dummy_model
import psp.models.psp_dcblstm_model as psp_dcblstm_model
import psp.models.psp_dculstm_model as psp_dculstm_model
import psp.models.auxiliary_models.psp_brnn_model as psp_brnn_model
import psp.models.auxiliary_models.psp_cnn_model as psp_cnn_model
import psp.models.auxiliary_models.psp_dcbgru_model as psp_dcbgru_model
import psp.models.auxiliary_models.psp_dcugru_model as psp_dcugru_model
import psp.models.auxiliary_models.psp_dnn_model as psp_dnn_model
import psp.models.auxiliary_models.psp_rnn_model as psp_rnn_model

#Test Case for testing dummy model
class DummyModelTests(unittest.TestCase):

    def setUp(self):
        """ Importing the required config files and model parameters. """

        #open JSON config file in config folder
        with open(os.path.join("config","dummy.json")) as f:
            self.dummy_model_params = json.load(f)

        #build model
        self.dummy_model_ = dummy_model.build_model(self.dummy_model_params["model_parameters"][0])

#1.)
    def test_shapes(self):
        """ Testing model shapes. """

        self.assertEqual(self.dummy_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dummy_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dummy_model_.get_layer("embedding").output_shape, (None,700,21))
        self.assertEqual(self.dummy_model_.get_layer("conv1").output_shape, (None,700,32))
        self.assertEqual(self.dummy_model_.get_layer("dense1").output_shape, (None,700,600))
        self.assertEqual(self.dummy_model_.get_layer("main_output").output_shape, (None,700,8))

#2.)
    def test_layer_types(self):
        """ Testing layer data types. """

        self.assertTrue(self.dummy_model_.get_layer('main_input').__class__.__name__ == "InputLayer")
        self.assertTrue(self.dummy_model_.get_layer('aux_input').__class__.__name__ == "InputLayer")
        self.assertTrue(self.dummy_model_.get_layer('conv1').__class__.__name__ == "Conv1D")
        self.assertTrue(self.dummy_model_.get_layer('dense1').__class__.__name__ == "Dense")
        self.assertTrue(self.dummy_model_.get_layer('main_output').__class__.__name__ == "Dense")

        self.assertEqual(self.dummy_model_.__class__.__name__,'Functional','')
        self.assertEqual(self.dummy_model_._name, 'dummy_model','')
        self.assertTrue(self.dummy_model_.compute_dtype=="float32")

#3.)
    @unittest.skip("Number of parameters of models is dynamic due to the implementation of the config files.")
    def test_trainable_params(self):
        """ Testing number of trainable & non-trainable parameters. """

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dummy_model_)
        self.assertTrue((trainable_params)==29113)
        self.assertTrue((non_trainable_params)==0)
        self.assertTrue((total_params)==29113)

    def tearDown(self):
        """ Deleting dummy model from memory. """
        del self.dummy_model_

#Test Case for testing DCBLSTM model
class DCBLSTMModelTests(unittest.TestCase):

    def setUp(self):
        """ Importing the required config files and model parameters. """

        #open JSON config file in config folder
        with open(os.path.join("config","dcblstm.json")) as f:
            self.dcblstm_model_params = json.load(f)

        #build model
        self.dcblstm_model_ = psp_dcblstm_model.build_model(self.dcblstm_model_params["model_parameters"][0])

#1.)
    def test_shapes(self):
        """ Testing model shapes. """

        self.assertEqual(self.dcblstm_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dcblstm_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dcblstm_model_.get_layer("embedding").output_shape, (None,700,21))

        self.assertEqual(self.dcblstm_model_.get_layer("conv1").output_shape, (None,700,32))
        self.assertEqual(self.dcblstm_model_.get_layer("batchNorm_1").output_shape, (None,700,32))
        self.assertEqual(self.dcblstm_model_.get_layer("conv2").output_shape, (None,700,64))
        self.assertEqual(self.dcblstm_model_.get_layer("batchNorm_2").output_shape, (None,700,64))
        self.assertEqual(self.dcblstm_model_.get_layer("conv3").output_shape, (None,700,128))
        self.assertEqual(self.dcblstm_model_.get_layer("batchNorm_3").output_shape, (None,700,128))

        self.assertEqual(self.dcblstm_model_.get_layer("blstm1").output_shape, (None,700,500))
        self.assertEqual(self.dcblstm_model_.get_layer("blstm2").output_shape, (None,700,500))

        self.assertEqual(self.dcblstm_model_.get_layer("dense1").output_shape, (None,700,600))
        self.assertEqual(self.dcblstm_model_.get_layer("dense2").output_shape, (None,700,300))
        self.assertEqual(self.dcblstm_model_.get_layer("main_output").output_shape, (None,700,8))

#2.)
    def test_layer_types(self):
        """ Testing model layer data types. """

        self.assertTrue(self.dcblstm_model_.get_layer('main_input').__class__.__name__ == "InputLayer")
        self.assertTrue(self.dcblstm_model_.get_layer('aux_input').__class__.__name__ == "InputLayer")

        self.assertTrue(self.dcblstm_model_.get_layer('conv1').__class__.__name__ == "Conv1D")
        self.assertTrue(self.dcblstm_model_.get_layer('conv2').__class__.__name__ == "Conv1D")
        self.assertTrue(self.dcblstm_model_.get_layer('conv3').__class__.__name__ == "Conv1D")

        self.assertTrue(self.dcblstm_model_.get_layer('blstm1').__class__.__name__ == "Bidirectional")
        self.assertTrue(self.dcblstm_model_.get_layer('blstm2').__class__.__name__ == "Bidirectional")

        self.assertTrue(self.dcblstm_model_.get_layer('dense1').__class__.__name__ == "TimeDistributed")
        self.assertTrue(self.dcblstm_model_.get_layer('dense2').__class__.__name__ == "TimeDistributed")
        self.assertTrue(self.dcblstm_model_.get_layer('main_output').__class__.__name__ == "TimeDistributed")

        self.assertEqual(self.dcblstm_model_.__class__.__name__,'Functional','')
        self.assertEqual(self.dcblstm_model_._name, 'psp_dcblstm_model','')
        self.assertTrue(self.dcblstm_model_.compute_dtype=="float32")

#3.)
    def test_paramater_values(self):
        """ Testing model layer parameter values. """

        self.assertEqual(len(self.dcblstm_model_.layers),23,'')

        #test conv layers
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][4]['config']['filters'],32)
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][4]['config']['kernel_size']),"(3,)")
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][4]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][4]['config']['strides']),"(1,)")
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][5]['config']['filters'],64)
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][5]['config']['kernel_size']),"(3,)")
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][5]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][5]['config']['strides']),"(1,)")
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][6]['config']['filters'],128)
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][6]['config']['kernel_size']),"(1,)")
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][6]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dcblstm_model_.get_config()['layers'][6]['config']['strides']),"(1,)")

        #test batch norm layers
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][7]['config']['epsilon'],0.001)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][8]['config']['momentum'],0.99)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][9]['config']['scale'],1)

        #test dropout layers
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][10]['config']['rate'],0.3)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][11]['config']['rate'],0.3)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][12]['config']['rate'],0.3)

        #test lstm layers
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][14]['config']['layer']['config']['units'],250)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][14]['config']['layer']['config']['dropout'],0.5)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][14]['config']['layer']['config']['recurrent_dropout'],0.5)
        self.assertTrue(self.dcblstm_model_.get_config()['layers'][14]['config']['layer']['config']['return_sequences'],'')
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][15]['config']['layer']['config']['units'],250)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][15]['config']['layer']['config']['dropout'],0.5)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][15]['config']['layer']['config']['recurrent_dropout'],0.5)
        self.assertTrue(self.dcblstm_model_.get_config()['layers'][15]['config']['layer']['config']['return_sequences'],'')

        #test dense layers
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][18]['config']['layer']['config']['units'],600)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][18]['config']['layer']['config']['activation'],'relu')
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][18]['class_name'],'TimeDistributed')

        self.assertEqual(self.dcblstm_model_.get_config()['layers'][20]['config']['layer']['config']['units'],300)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][20]['config']['layer']['config']['activation'],'relu')
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][20]['class_name'],'TimeDistributed')

        self.assertEqual(self.dcblstm_model_.get_config()['layers'][22]['config']['layer']['config']['units'],8)
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][22]['config']['layer']['config']['activation'],'softmax')
        self.assertEqual(self.dcblstm_model_.get_config()['layers'][22]['class_name'],'TimeDistributed')

#4.)
    def test_trainable_params(self):
        """ Testing number of trainable & non-trainable parameters. """

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dcblstm_model_)
        self.assertTrue((trainable_params)==3388293)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==3388741)

    def tearDown(self):
        """ Deleting model from memory. """
        del self.dcblstm_model_

#Test Case for testing DCULSTM model
class DCULSTMModelTests(unittest.TestCase):

    def setUp(self):
        """ Importing the required config files and model parameters. """

        #open JSON config file in config folder
        with open(os.path.join("config","dculstm.json")) as f:
            self.dculstm_model_params = json.load(f)

        #build model
        self.dculstm_model_ = psp_dculstm_model.build_model(self.dculstm_model_params["model_parameters"][0])

#1.)
    def test_shapes(self):
        """ Testing model shapes. """

        self.assertEqual(self.dculstm_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dculstm_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dculstm_model_.get_layer("embedding").output_shape, (None,700,21))

        self.assertEqual(self.dculstm_model_.get_layer("conv1").output_shape, (None,700,32))
        self.assertEqual(self.dculstm_model_.get_layer("batchNorm_1").output_shape, (None,700,32))
        self.assertEqual(self.dculstm_model_.get_layer("conv2").output_shape, (None,700,64))
        self.assertEqual(self.dculstm_model_.get_layer("batchNorm_2").output_shape, (None,700,64))
        self.assertEqual(self.dculstm_model_.get_layer("conv3").output_shape, (None,700,128))
        self.assertEqual(self.dculstm_model_.get_layer("batchNorm_3").output_shape, (None,700,128))

        self.assertEqual(self.dculstm_model_.get_layer("lstm1").output_shape, (None,700,250))
        self.assertEqual(self.dculstm_model_.get_layer("lstm2").output_shape, (None,700,250))
        self.assertEqual(self.dculstm_model_.get_layer("lstm3").output_shape, (None,700,250))

        self.assertEqual(self.dculstm_model_.get_layer("dense1").output_shape, (None,700,600))
        self.assertEqual(self.dculstm_model_.get_layer("dense2").output_shape, (None,700,300))
        self.assertEqual(self.dculstm_model_.get_layer("main_output").output_shape, (None,700,8))

#2.)
    def test_layer_types(self):
        """ Testing model layer data types. """

        self.assertTrue(self.dculstm_model_.get_layer('main_input').__class__.__name__ == "InputLayer")
        self.assertTrue(self.dculstm_model_.get_layer('aux_input').__class__.__name__ == "InputLayer")

        self.assertTrue(self.dculstm_model_.get_layer('conv1').__class__.__name__ == "Conv1D")
        self.assertTrue(self.dculstm_model_.get_layer('conv2').__class__.__name__ == "Conv1D")
        self.assertTrue(self.dculstm_model_.get_layer('conv3').__class__.__name__ == "Conv1D")

        self.assertTrue(self.dculstm_model_.get_layer('lstm1').__class__.__name__ == "LSTM")
        self.assertTrue(self.dculstm_model_.get_layer('lstm2').__class__.__name__ == "LSTM")
        self.assertTrue(self.dculstm_model_.get_layer('lstm3').__class__.__name__ == "LSTM")

        self.assertTrue(self.dculstm_model_.get_layer('dense1').__class__.__name__ == "Dense")
        self.assertTrue(self.dculstm_model_.get_layer('dense2').__class__.__name__ == "Dense")
        self.assertTrue(self.dculstm_model_.get_layer('main_output').__class__.__name__ == "Dense")

        self.assertEqual(self.dculstm_model_.__class__.__name__,'Functional','')
        self.assertEqual(self.dculstm_model_._name, 'psp_dculstm_model','')
        self.assertTrue(self.dculstm_model_.compute_dtype=="float32")

#3.)
    def test_paramater_values(self):
        """ Testing model layer parameter values. """

        #test conv layers
        self.assertEqual(len(self.dculstm_model_.layers),24,'')
        self.assertEqual(self.dculstm_model_.get_config()['layers'][4]['config']['filters'],32)
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][4]['config']['kernel_size']),"(3,)")
        self.assertEqual(self.dculstm_model_.get_config()['layers'][4]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][4]['config']['strides']),"(1,)")
        self.assertEqual(self.dculstm_model_.get_config()['layers'][5]['config']['filters'],64)
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][5]['config']['kernel_size']),"(3,)")
        self.assertEqual(self.dculstm_model_.get_config()['layers'][5]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][5]['config']['strides']),"(1,)")
        self.assertEqual(self.dculstm_model_.get_config()['layers'][6]['config']['filters'],128)
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][6]['config']['kernel_size']),"(3,)")
        self.assertEqual(self.dculstm_model_.get_config()['layers'][6]['config']['kernel_initializer']['class_name'],"GlorotUniform")
        self.assertEqual(str(self.dculstm_model_.get_config()['layers'][6]['config']['strides']),"(1,)")

        #test batch norm layers
        self.assertEqual(self.dculstm_model_.get_config()['layers'][7]['config']['epsilon'],0.001)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][8]['config']['momentum'],0.99)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][9]['config']['scale'],1)

        #test dropout layers
        self.assertEqual(self.dculstm_model_.get_config()['layers'][10]['config']['rate'],0.3)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][11]['config']['rate'],0.3)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][12]['config']['rate'],0.3)

        #test lstm layers
        self.assertEqual(self.dculstm_model_.get_config()['layers'][14]['config']['units'],250)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][14]['config']['dropout'],0.5)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][14]['config']['recurrent_dropout'],0.5)
        self.assertTrue(self.dculstm_model_.get_config()['layers'][14]['config']['return_sequences'],'')
        self.assertEqual(self.dculstm_model_.get_config()['layers'][15]['config']['units'],250)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][15]['config']['dropout'],0.5)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][15]['config']['recurrent_dropout'],0.5)
        self.assertTrue(self.dculstm_model_.get_config()['layers'][15]['config']['return_sequences'],'')
        self.assertEqual(self.dculstm_model_.get_config()['layers'][16]['config']['units'],250)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][16]['config']['dropout'],0.5)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][16]['config']['recurrent_dropout'],0.5)
        self.assertTrue(self.dculstm_model_.get_config()['layers'][16]['config']['return_sequences'],'')

        #test dense layers
        self.assertEqual(self.dculstm_model_.get_config()['layers'][19]['config']['units'],600)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][19]['config']['activation'],'relu')
        self.assertEqual(self.dculstm_model_.get_config()['layers'][21]['config']['units'],300)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][21]['config']['activation'],'relu')
        self.assertEqual(self.dculstm_model_.get_config()['layers'][23]['config']['units'],8)
        self.assertEqual(self.dculstm_model_.get_config()['layers'][23]['config']['activation'],'softmax')

#4.)
    @unittest.skip("Number of parameters of models is dynamic due to the implementation of the config files.")
    def test_trainable_params(self):
        """ Testing number of trainable & non-trainable parameters. """

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dculstm_model_)
        self.assertTrue((trainable_params)==2274045)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==2274493)

    def tearDown(self):
        """ Deleting model from memory. """
        del self.dculstm_model_

#Test Case for testing Auxillary models
class AuxillaryModelTests(unittest.TestCase):

    def setUp(self):
        """ Importing the required config files and model parameters. """

        brnn_config_file = os.path.join("config","brnn.json")
        cnn_config_file = os.path.join("config","cnn.json")
        dcbgru_config_file = os.path.join("config","dcbgru.json")
        dcugru_config_file = os.path.join("config","dcugru.json")
        dnn_config_file = os.path.join("config","dnn.json")
        rnn_config_file = os.path.join("config","rnn.json")

        #open JSON config file in config folder
        with open(brnn_config_file) as f:
            self.brnn_model_params = json.load(f)
        with open(cnn_config_file) as f:
            self.cnn_model_params = json.load(f)
        with open(dcbgru_config_file) as f:
            self.dcbgru_model_params = json.load(f)
        with open(dcugru_config_file) as f:
            self.dcugru_model_params = json.load(f)
        with open(dnn_config_file) as f:
            self.dnn_model_params = json.load(f)
        with open(rnn_config_file) as f:
            self.rnn_model_params = json.load(f)

        #build models
        self.brnn_model_ = psp_brnn_model.build_model(self.brnn_model_params["model_parameters"][0])
        self.cnn_model_ = psp_cnn_model.build_model(self.cnn_model_params["model_parameters"][0])
        self.dcbgru_model_ = psp_dcbgru_model.build_model(self.dcbgru_model_params["model_parameters"][0])
        self.dcugru_model_ = psp_dcugru_model.build_model(self.dcugru_model_params["model_parameters"][0])
        self.dnn_model_ = psp_dnn_model.build_model(self.dnn_model_params["model_parameters"][0])
        self.rnn_model_ = psp_rnn_model.build_model(self.rnn_model_params["model_parameters"][0])

#1.)
    def test_layers(self):
        """ Testing model layer shapes. """

        self.assertEqual(self.brnn_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.brnn_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.brnn_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(self.cnn_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.cnn_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.cnn_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(self.dcbgru_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dcbgru_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dcbgru_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(self.dcugru_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dcugru_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dcugru_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(self.dnn_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.dnn_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.dnn_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(self.rnn_model_.get_layer("main_input").output_shape, [(None,700)])
        self.assertEqual(self.rnn_model_.get_layer("aux_input").output_shape, [(None,700,21)])
        self.assertEqual(self.rnn_model_.get_layer("main_output").output_shape, (None,700,8))

        self.assertEqual(len(self.brnn_model_.layers),20,'')
        self.assertEqual(len(self.cnn_model_.layers),18,'')
        self.assertEqual(len(self.dcbgru_model_.layers),22,'')
        self.assertEqual(len(self.dcugru_model_.layers),24,'')
        self.assertEqual(len(self.dnn_model_.layers),17,'')
        self.assertEqual(len(self.rnn_model_.layers),24,'')

        self.assertEqual(self.brnn_model_._name, 'psp_brnn_model','')
        self.assertEqual(self.cnn_model_._name, 'psp_cnn_model','')
        self.assertEqual(self.dcbgru_model_._name, 'psp_dcbgru_model','')
        self.assertEqual(self.dcugru_model_._name, 'psp_dcugru_model','')
        self.assertEqual(self.dnn_model_._name, 'psp_dnn_model','')
        self.assertEqual(self.rnn_model_._name, 'psp_rnn_model','')

#2.)
    def test_params(self):
        """ Testing number of trainable & non-trainable parameters. """

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.brnn_model_)
        self.assertTrue((trainable_params)==1012245)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==1012693)

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.cnn_model_)
        self.assertTrue((trainable_params)==99245)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==99693)

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dcbgru_model_)
        self.assertTrue((trainable_params)==2789045)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==2789493)

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dcugru_model_)
        self.assertTrue((trainable_params)==1907045)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==1907493)

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.dnn_model_)
        self.assertTrue((trainable_params)==262901)
        self.assertTrue((non_trainable_params)==0)
        self.assertTrue((total_params)==262901)

        trainable_params, non_trainable_params, total_params = get_trainable_parameters(self.rnn_model_)
        self.assertTrue((trainable_params)==1166295)
        self.assertTrue((non_trainable_params)==448)
        self.assertTrue((total_params)==1166743)

    def tearDown(self):
        """ Deleting models from memory. """

        del self.brnn_model_
        del self.cnn_model_
        del self.dcbgru_model_
        del self.dcugru_model_
        del self.dnn_model_
        del self.rnn_model_

if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)
