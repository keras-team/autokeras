import tensorflow as tf
from autokeras.hyperparameters import HyperParameters


def test_hyperparameters():
    hp = HyperParameters()
    with tf.name_scope('abc'):
        hp.Choice('num_layers', [1, 2, 3], default=1)
    assert 'abc/num_layers' in hp.values
