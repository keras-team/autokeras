import kerastuner
import pytest
import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module

import autokeras as ak
from autokeras.hypermodel import graph as graph_module
from tests import common


def test_set_hp():
    input_node = ak.Input((32,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    head = ak.RegressionHead()
    head.output_shape = (1,)
    output_node = head(output_node)

    graph = graph_module.HyperGraph(
        input_node,
        output_node,
        override_hps=[hp_module.Choice('dense_block_1/num_layers', [6], default=6)])
    hp = kerastuner.HyperParameters()
    plain_graph = graph.hyper_build(hp)
    plain_graph.build_keras_graph().build(hp)

    for single_hp in hp.space:
        if single_hp.name == 'dense_block_1/num_layers':
            if not (len(single_hp.values) == 1 and single_hp.values[0] == 6):
                raise AssertionError()
            return
    raise AssertionError()


def test_input_output_disconnect():
    input_node1 = ak.Input()
    output_node = input_node1
    _ = ak.DenseBlock()(output_node)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.HyperGraph(input_node1, output_node)
    if 'Inputs and outputs not connected.' not in str(info.value):
        raise AssertionError()


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
        graph_module.HyperGraph([input_node1, input_node2], output_node)
    if 'The network has a cycle.' not in str(info.value):
        raise AssertionError()


def test_input_missing():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead()(output_node)

    with pytest.raises(ValueError) as info:
        graph_module.HyperGraph(input_node1, output_node)
    if 'A required input is missing for HyperModel' not in str(info.value):
        raise AssertionError()


def test_graph_basics():
    input_node = ak.Input(shape=(30,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead(output_shape=(1,))(output_node)

    graph = graph_module.PlainGraph(input_node, output_node)
    model = graph.build_keras_graph().build(kerastuner.HyperParameters())
    if not (model.input_shape == (None, 30) and model.output_shape == (None, 1)):
        raise AssertionError()


def test_merge():
    input_node1 = ak.Input(shape=(30,))
    input_node2 = ak.Input(shape=(40,))
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    output_node = ak.RegressionHead(output_shape=(1,))(output_node)

    graph = graph_module.PlainGraph([input_node1, input_node2],
                                    output_node)
    model = graph.build_keras_graph().build(kerastuner.HyperParameters())
    if not (model.input_shape == [(None, 30), (None, 40)] and
            model.output_shape == (None, 1)):
        raise AssertionError()


def test_preprocessing():
    input_shape = (33,)
    output_shape = (1,)
    x_train_1 = common.generate_data(
        num_instances=100,
        shape=input_shape,
        dtype='dataset')
    x_train_2 = common.generate_data(
        num_instances=100,
        shape=input_shape,
        dtype='dataset')
    y_train = common.generate_data(
        num_instances=100,
        shape=output_shape,
        dtype='dataset')
    dataset = tf.data.Dataset.zip(((x_train_1, x_train_2), y_train))

    input_node1 = ak.Input(shape=input_shape)
    temp_node1 = ak.Normalization()(input_node1)
    output_node1 = ak.DenseBlock()(temp_node1)

    output_node3 = ak.Normalization()(temp_node1)
    output_node3 = ak.DenseBlock()(output_node3)

    input_node2 = ak.Input(shape=input_shape)
    output_node2 = ak.Normalization()(input_node2)
    output_node2 = ak.DenseBlock()(output_node2)

    output_node = ak.Merge()([output_node1, output_node2, output_node3])
    output_node = ak.RegressionHead()(output_node)

    graph = graph_module.PlainGraph([input_node1, input_node2],
                                    output_node)
    graph.build_preprocess_graph().preprocess(
        dataset=dataset,
        validation_data=dataset,
        fit=True)
