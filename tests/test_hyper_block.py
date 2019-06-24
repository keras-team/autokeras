import pytest

from autokeras.auto.auto_model import *
from autokeras import const

from autokeras.hypermodel.hyper_block import *
from autokeras.hypermodel.hyper_head import ClassificationHead
from autokeras.hypermodel.hyper_head import RegressionHead
from autokeras.hypermodel.hyper_head import SequenceHead

from autokeras.hypermodel.hyper_node import Input
from autokeras.hyperparameters import HyperParameters

import numpy as np
import tensorflow as tf


def test_xception_block():
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.randint(10, size=100)
    y_train = tf.keras.utils.to_categorical(y_train)

    input_node = Input()
    output_node = input_node
    output_node = XceptionBlock()(output_node)
    output_node = ClassificationHead()(output_node)

    input_node.shape = (32, 32, 3)
    output_node[0].shape = (10,)

    graph = GraphAutoModel(input_node, output_node)
    model = graph.build(HyperParameters())
    model.fit(x_train, y_train, epochs=1, batch_size=100, verbose=False)
    result = model.predict(x_train)

    assert result.shape == (100, 10)


def test_rnn_block():

    x_train = np.random.rand(100, 32, 20, 10)
    y_train = np.random.rand(100)

    input_node = Input()
    output_node = input_node
    output_node = RNNBlock()(output_node)
    output_node = RegressionHead()(output_node)

    auto_model = GraphAutoModel(input_node, output_node)
    const.Constant.NUM_TRAILS = 2
    auto_model.fit(x_train, y_train, epochs=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 1)


def test_seq2seq():
    x_train = np.random.rand(100, 32, 20)

    input_node = Input()
    output_node = input_node
    output_node = RNNBlock(seq2seq=True)(output_node)
    output_node = SequenceHead()(output_node)

    auto_model = GraphAutoModel(input_node, output_node)
    const.Constant.NUM_TRAILS = 2
    auto_model.fit(x_train, x_train, epochs=2)
    result = auto_model.predict(x_train)

    assert result.shape == (100, 32,20)

