import kerastuner
from kerastuner.engine import hyperparameters as hp_module
import pytest

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


def test_input_output_disconnect():
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        ak.hypermodel.graph.GraphHyperModel(input_node1, output_node)
    assert 'Inputs and outputs not connected.' in str(info.value)


def test_hyper_graph_cycle():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    head = ak.RegressionHead()
    output_node = head(output_node)
    head.outputs = output_node1

    with pytest.raises(ValueError) as info:
        ak.hypermodel.graph.GraphHyperModel([input_node1, input_node2],
                                            output_node)
    assert 'The network has a cycle.' in str(info.value)


def test_input_missing():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        ak.hypermodel.graph.GraphHyperModel(input_node1, output_node)
    assert 'A required input is missing for HyperModel' in str(info.value)
