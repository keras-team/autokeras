import kerastuner
from kerastuner.engine import hyperparameters as hp_module

import autokeras as ak


def test_set_hp():
    input_node = ak.Input((32,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    head = ak.RegressionHead()
    head.output_shape = (1,)
    output_node = head(output_node)

    graph = ak.hypermodel.graph.HyperBuiltGraphHyperModel(input_node, output_node)
    hp = kerastuner.HyperParameters()
    graph.set_hps([hp_module.Choice('dense_block_1/num_layers', [6], default=6)])
    graph.build(hp)

    for single_hp in hp.space:
        if single_hp.name == 'dense_block_1/num_layers':
            assert len(single_hp.values) == 1
            assert single_hp.values[0] == 6
            return
    assert False
