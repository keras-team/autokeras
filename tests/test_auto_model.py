import numpy as np
from autokeras.auto.auto_model import *
from autokeras.hypermodel.hyper_block import MlpBlock, HierarchicalHyperParameters
from autokeras.hypermodel.hyper_head import RegressionHead
from autokeras.hypermodel.hyper_node import Input


def test_auto_model_basic():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node = MlpBlock()(output_node)
    output_node = RegressionHead()(output_node)

    input_node.shape = (32,)
    output_node[0].shape = (1,)

    auto_model = AutoModel(input_node, output_node)
    auto_model.compile(loss='mean_squared_error', optimizer='adam')
    auto_model.fit(x_train, y_train, trails=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 1)
