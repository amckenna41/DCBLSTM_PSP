
##########################
### Tests for Models ###
##########################

import unittest
# import models.psp_dcblstm_model as psp_dcblstm_model
# import models.psp_dculstm_model as psp_dculstm_model
from models import *
from models.auxiliary_models import *

def tests_tensorflow():

    # protein_seq = tf.constant(700)
    #
    # with tf.Session() as sess:
    #     main_input = tf.Variable()

    pass
def test_model_layer_shapes():
    """
    Testing Model Layer Shapes
    Args:

    Returns:

    """
    model = models.psp_cdblstm_model.build_model()

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

    model = psp_cdulstm_model.build_model()

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
    Description:
        Testing Model Layer Types
    Args:
        None
    Returns:
        None

    """

    ##iterate through all models checking respective layers names and types ##
    model = models.psp_cdblstm_model.build_model()

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

    model = psp_cdulstm_model.build_model()

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

def run_model_tests():

    test_model_layer_shapes()
    print("Model layer shape tests passed")

    test_model_layer_types()
    print("Model layer shape types passed")
    print('')
    print('All Model tests passed')

if __name__ == '__main__':

    run_model_tests()
