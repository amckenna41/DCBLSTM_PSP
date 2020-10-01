import unittest
from psp_blstm_model import *
from psp_ulstm_model import *

#testing model's layer shapes
def test_model_layer_shapes():

    model = psp_blstm_model.build_model()

    #assertions for model layer shapes
    assert(model.get_layer("main_input").output_shape == (None,700))
    assert(model.get_layer("aux_input").output_shape == (None,700,21))

    assert(model.get_layer("Conv1D_1").output_shape == (None,700,16))
    assert(model.get_layer("Conv1D_2").output_shape == (None,700,32))
    assert(model.get_layer("Conv1D_3").output_shape == (None,700,64))

    assert(model.get_layer("MaxPool_1").output_shape == (None,700,16))
    assert(model.get_layer("MaxPool_2").output_shape == (None,700,32))
    assert(model.get_layer("MaxPool_3").output_shape == (None,700,64))

    assert(model.get_layer("after_cnn_dense").output_shape == (None,700,600))
    assert(model.get_layer("after_rnn_dense").output_shape == (None,700,600))

#testing model layer types
def test_model_layer_types():

    model = psp_blstm_model.build_model()

    #assertions for model layer types
    assert(model.get_layer('main_input').__class__.__name__ == "InputLayer")
    assert(model.get_layer('aux_input').__class__.__name__ == "InputLayer")

    assert(model.get_layer('Conv1D_1').__class__.__name__ == "Conv1D")
    assert(model.get_layer('Conv1D_2').__class__.__name__ == "Conv1D")
    assert(model.get_layer('Conv1D_3').__class__.__name__ == "Conv1D")

    assert(model.get_layer('MaxPool_1').__class__.__name__ == "MaxPooling1D")
    assert(model.get_layer('MaxPool_2').__class__.__name__ == "MaxPooling1D")
    assert(model.get_layer('MaxPool_3').__class__.__name__ == "MaxPooling1D")

    assert(model.get_layer('after_cnn_dense').__class__.__name__ == "Dense")
    assert(model.get_layer('after_rnn_dense').__class__.__name__ == "Dense")
