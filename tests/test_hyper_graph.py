import numpy as np

from autokeras.hypermodel.hyper_block import MlpBlock, HierarchicalHyperParameters, Merge
from autokeras.hypermodel.hyper_head import RegressionHead
from autokeras.hypermodel.hyper_node import Input
from autokeras.hypermodel.hyper_graph import *


def test_hyper_graph_basic():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node = MlpBlock()(output_node)
    output_node = RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    graph = HyperGraph(input_node, output_node)
    model = graph.build(HierarchicalHyperParameters())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 1)


def test_merge():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node1 = MlpBlock()(output_node)
    output_node2 = MlpBlock()(output_node)
    output_node = Merge()([output_node1, output_node2])
    output_node = RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    graph = HyperGraph(input_node, output_node)
    model = graph.build(HierarchicalHyperParameters())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 1)


def test_input_output_disconnect():
    pass


def test_hyper_graph_cycle():
    pass


def test_input_missing():
    pass
