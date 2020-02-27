from unittest import mock

import kerastuner
import pytest

import autokeras as ak
from autokeras import graph as graph_module
from autokeras.hypermodels import basic
from tests import utils


def test_type_error_for_call():
    block = basic.ConvBlock()
    with pytest.raises(TypeError) as info:
        block(block)
    assert 'Expect the inputs to layer' in str(info.value)


@mock.patch('autokeras.hypermodels.basic.resnet.HyperResNet.__init__')
@mock.patch('autokeras.hypermodels.basic.resnet.HyperResNet.build')
def test_resnet_block(init, build):
    input_shape = (32, 32, 3)
    block = basic.ResNetBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('version', hp)
    assert utils.name_in_hps('pooling', hp)
    assert init.called
    assert build.called


@mock.patch('autokeras.hypermodels.basic.xception.HyperXception.__init__')
@mock.patch('autokeras.hypermodels.basic.xception.HyperXception.build')
def test_xception_block(init, build):
    input_shape = (32, 32, 3)
    block = basic.XceptionBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('activation', hp)
    assert utils.name_in_hps('initial_strides', hp)
    assert utils.name_in_hps('num_residual_blocks', hp)
    assert utils.name_in_hps('pooling', hp)
    assert init.called
    assert build.called


def test_conv_block():
    input_shape = (32, 32, 3)
    block = basic.ConvBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('kernel_size', hp)
    assert utils.name_in_hps('num_blocks', hp)
    assert utils.name_in_hps('separable', hp)


def test_rnn_block():
    input_shape = (32, 10)
    block = basic.RNNBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('bidirectional', hp)
    assert utils.name_in_hps('layer_type', hp)
    assert utils.name_in_hps('num_layers', hp)


def test_dense_block():
    input_shape = (32,)
    block = basic.DenseBlock()
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('num_layers', hp)
    assert utils.name_in_hps('use_batchnorm', hp)


def test_embedding_block():
    input_shape = (32,)
    block = basic.Embedding()
    block.max_features = 100
    hp = kerastuner.HyperParameters()

    block = graph_module.deserialize(graph_module.serialize(block))
    block.build(hp, ak.Input(shape=input_shape).build())

    assert utils.name_in_hps('pretraining', hp)
    assert utils.name_in_hps('embedding_dim', hp)
