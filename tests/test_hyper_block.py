import pytest

from autokeras.auto.auto_model import *

from autokeras.hypermodel.hyper_block import *
from autokeras.hypermodel.hyper_head import ClassificationHead
from autokeras.hypermodel.hyper_node import Input

import numpy as np
import tensorflow as tf


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_hyper_block')


def test_xception_block(tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.randint(10, size=100)
    y_train = tf.keras.utils.to_categorical(y_train)

    input_node = Input()
    output_node = input_node
    output_node = XceptionBlock()(output_node)
    output_node = ClassificationHead()(output_node)

    input_node.shape = (32, 32, 3)
    output_node[0].shape = (10,)

    graph = GraphAutoModel(input_node, output_node, directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 10)
