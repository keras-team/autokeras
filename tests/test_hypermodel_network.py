from autokeras.hypermodel.hyper_block import HierarchicalHyperParameters
from autokeras.hypermodel.hypermodel_network import *


def test_connected_hyperparameters():
    hp = HierarchicalHyperParameters()
    with tf.name_scope('abc'):
        hp.Choice('num_layers', [1, 2, 3], default=1)
    assert 'abc/num_layers' in hp.values

