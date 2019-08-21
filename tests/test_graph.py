import kerastuner
import numpy as np
from kerastuner.engine import hyperparameters as hp_module

import autokeras as ak


def test_set_hp():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input((32,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    head = ak.RegressionHead()
    head.output_shape = (1,)
    output_node = head(output_node)

    graph = ak.hypermodel.graph.GraphHyperModel(input_node, output_node)
    hp = kerastuner.HyperParameters()
    graph.set_hps([hp_module.Choice('num_layers', [6], default=6)])
    with hp.name_scope('dense_block_1'):
        graph.build(hp)

    for single_hp in hp.space:
        if single_hp.name == 'dense_block_1/num_layers':
            assert len(single_hp.values) == 1
            assert single_hp.values[0] == 6
            return
    assert False
