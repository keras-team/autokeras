import numpy as np

from autokeras.auto.auto_model import GraphAutoModel
from autokeras.hypermodel.hyper_head import ClassificationHead
from autokeras.hypermodel.hyper_node import Input
from autokeras.hypermodel.resnet import *


def test_resnet_block():
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 10)

    input_node = Input()
    output_node = input_node
    output_node = ResNetBlock()(output_node)
    output_node = ClassificationHead()(output_node)

    auto_model = GraphAutoModel(input_node, output_node)
    auto_model.fit(x_train, y_train, trials=1, epochs=1)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 10)
