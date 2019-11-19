from unittest import mock

import kerastuner
import pytest

import autokeras as ak
from autokeras.hypermodel import block as block_module
from tests import common


def test_type_error_for_call():
    block = block_module.ConvBlock()
    with pytest.raises(TypeError) as info:
        block(block)
    if 'Expect the inputs to layer' not in str(info.value):
        raise AssertionError()


@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.__init__')
@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.build')
def test_resnet_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.ResNetBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('version', hp) and common.name_in_hps('pooling', hp)
            and init.called and build.called):
        raise AssertionError()


@mock.patch('autokeras.hypermodel.block.xception.HyperXception.__init__')
@mock.patch('autokeras.hypermodel.block.xception.HyperXception.build')
def test_xception_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.XceptionBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('activation', hp) and
            common.name_in_hps('initial_strides', hp) and
            common.name_in_hps('num_residual_blocks', hp) and
            common.name_in_hps('pooling', hp) and init.called and build.called):
        raise AssertionError()


def test_conv_block():
    input_shape = (32, 32, 3)
    block = block_module.ConvBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('kernel_size', hp) and
            common.name_in_hps('num_blocks', hp) and
            common.name_in_hps('separable', hp)):
        raise AssertionError()


def test_rnn_block():
    input_shape = (32, 10)
    block = block_module.RNNBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('bidirectional', hp) and
            common.name_in_hps('layer_type', hp) and
            common.name_in_hps('num_layers', hp)):
        raise AssertionError()


def test_dense_block():
    input_shape = (32,)
    block = block_module.DenseBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('num_layers', hp) and
            common.name_in_hps('use_batchnorm', hp)):
        raise AssertionError()


def test_merge():
    input_shape_1 = (32,)
    input_shape_2 = (4, 8)
    block = block_module.Merge()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, [ak.Input(shape=input_shape_1).build(),
                     ak.Input(shape=input_shape_2).build()])

    if not common.name_in_hps('merge_type', hp):
        raise AssertionError()


def test_temporal_reduction():
    input_shape = (32, 10)
    block = block_module.TemporalReduction()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not common.name_in_hps('reduction_type', hp):
        raise AssertionError()


def test_spatial_reduction():
    input_shape = (32, 32, 3)
    block = block_module.SpatialReduction()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not common.name_in_hps('reduction_type', hp):
        raise AssertionError()


def test_embedding_block():
    input_shape = (32,)
    block = block_module.EmbeddingBlock()
    block.max_features = 100
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    if not (common.name_in_hps('pretraining', hp) and
            common.name_in_hps('embedding_dim', hp)):
        raise AssertionError()
