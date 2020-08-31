import unittest
from psp_3conv_bgru_model import *
#make models and test output, input shapes
#first and last layer are correct

def test_model_input():

    model =
    pass
model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
