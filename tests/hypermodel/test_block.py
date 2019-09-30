from unittest import mock

import kerastuner
import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from tests import common


def test_resnet_block():
    input_shape = (32, 32, 3)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape)
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes)

    input_node = ak.Input(shape=input_shape)
    output_node = input_node
    output_node = ak.ResNetBlock()(output_node)
    output_node = ak.ClassificationHead(output_shape=(num_classes,))(output_node)

    result = common.fit_predict_with_graph(input_node, output_node, x, y)
    assert result.shape == (num_instances, num_classes)


def test_xception_block():
    input_shape = (32, 32, 3)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape)
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes)

    input_node = ak.Input(shape=input_shape)
    output_node = input_node
    output_node = ak.XceptionBlock()(output_node)
    output_node = ak.ClassificationHead(output_shape=(num_classes,))(output_node)

    result = common.fit_predict_with_graph(input_node, output_node, x, y)
    assert result.shape == (num_instances, num_classes)


def test_conv_block():
    input_shape = (32, 32, 3)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape)
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes)

    input_node = ak.Input(shape=input_shape)
    output_node = input_node
    output_node = ak.ConvBlock()(output_node)
    output_node = ak.ClassificationHead(output_shape=(num_classes,))(output_node)

    result = common.fit_predict_with_graph(input_node, output_node, x, y)
    assert result.shape == (num_instances, num_classes)


def test_rnn_block():
    input_shape = (32, 10)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape)
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes)

    input_node = ak.Input(shape=input_shape)
    output_node = input_node
    output_node = ak.RNNBlock()(output_node)
    output_node = ak.ClassificationHead(output_shape=(num_classes,))(output_node)

    result = common.fit_predict_with_graph(input_node, output_node, x, y)
    assert result.shape == (num_instances, num_classes)
