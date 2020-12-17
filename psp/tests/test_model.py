
##########################
### Tests for Models ###
##########################

import unittest
import models.psp_blstm_model as psp_blstm_model
import models.psp_ulstm_model as psp_ulstm_model

def test_model_layer_shapes():
    """
    Testing Model Layer Shapes
    Args:

    Returns:

    """
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

def test_model_layer_types():
    """
    Testing Model Layer Types
    Args:

    Returns:

    """

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

def run_tests():

    test_model_layer_shapes()
    print("Model layer shape tests passed")

    test_model_layer_types()
    print("Model layer shape types passed")
