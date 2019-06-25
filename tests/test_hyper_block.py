import kerastuner
import pytest

import numpy as np
import tensorflow as tf

from autokeras import const
from autokeras.auto import auto_model as am_module
from autokeras.hypermodel import hyper_node, hyper_block, hyper_head


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_hyper_block')


def test_xception_block(tmp_dir):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.randint(10, size=100)
    y_train = tf.keras.utils.to_categorical(y_train)

    input_node = hyper_node.Input()
    output_node = input_node
    output_node = hyper_block.XceptionBlock()(output_node)
    output_node = hyper_head.ClassificationHead()(output_node)

    input_node.shape = (32, 32, 3)
    output_node[0].shape = (10,)

    graph = am_module.GraphAutoModel(input_node, output_node, directory=tmp_dir)
    model = graph.build(kerastuner.HyperParameters())
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 10)


def test_rnn_block():
    x_train = np.random.rand(100, 32, 20, 10)
    y_train = np.random.rand(100)

    input_node = hyper_node.Input()
    output_node = input_node
    output_node = hyper_block.RNNBlock()(output_node)
    output_node = hyper_head.RegressionHead()(output_node)

    auto_model = am_module.GraphAutoModel(input_node, output_node)
    const.Constant.NUM_TRAILS = 2
    auto_model.fit(x_train, y_train, epochs=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 1)
