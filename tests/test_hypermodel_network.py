import numpy as np

from autokeras.hypermodel.hyper_block import MlpBlock, HierarchicalHyperParameters
from autokeras.hypermodel.hyper_head import RegressionHead
from autokeras.hypermodel.hyper_node import Input
from autokeras.hypermodel.hypermodel_network import *


def test_hyper_graph_basic():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)
    input_node = Input()
    output_node = input_node
    output_node = MlpBlock()(output_node)
    output_node = RegressionHead()(output_node)
    input_node.shape = (32,)
    graph = HyperGraph(input_node, output_node)
    model = graph.build(HierarchicalHyperParameters())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=5)
    result = model.predict(x_train)
    assert result.shape == (100,)


def test_hyper_graph_cycle():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)
    input_node = Input()
    output_node = input_node
    output_node = MlpBlock()(output_node)
    output_node = RegressionHead()(output_node)
    graph = HyperGraph(input_node, output_node)
    model = graph.build(HierarchicalHyperParameters())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=5)
