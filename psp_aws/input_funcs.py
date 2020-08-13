#train_input_fn, eval_input_fn, serving_input_fn functions are used buy SageMaker
#to help train, evaluate and serve the model


import tensorflow as tf

INPUT_TENSOR_NAME = "main_input" # Watch out, it needs to match the name of the first layer
BATCH_SIZE = 42

def serving_input_fn(hyperparameters):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(tf.float32, shape=[None, 5278, 700, 21])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)

# Be careful, for train_input_fn and eval_input_fn as well, the first argument must be named “training_dir”.
