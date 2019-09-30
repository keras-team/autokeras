from unittest import mock

import kerastuner
import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodel import block as block_module
from tests import common


def name_in_hps(hp_name, hp):
    return any([hp_name in name for name in hp.values])


@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.__init__')
@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.build')
def test_resnet_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.ResNetBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('version', hp)
    assert name_in_hps('pooling', hp)
    assert init.called
    assert build.called


@mock.patch('autokeras.hypermodel.block.xception.HyperXception.__init__')
@mock.patch('autokeras.hypermodel.block.xception.HyperXception.build')
def test_xception_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.XceptionBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('activation', hp)
    assert name_in_hps('initial_strides', hp)
    assert name_in_hps('num_residual_blocks', hp)
    assert name_in_hps('pooling', hp)
    assert init.called
    assert build.called


def test_conv_block():
    input_shape = (32, 32, 3)
    block = block_module.ConvBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('kernel_size', hp)
    assert name_in_hps('num_blocks', hp)
    assert name_in_hps('separable', hp)


def test_rnn_block():
    input_shape = (32, 10)
    block = block_module.RNNBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('bidirectional', hp)
    assert name_in_hps('layer_type', hp)
    assert name_in_hps('num_layers', hp)


def test_dense_block():
    input_shape = (32,)
    block = block_module.DenseBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('num_layers', hp)
    assert name_in_hps('use_batchnorm', hp)


def test_merge():
    input_shape_1 = (32,)
    input_shape_2 = (4, 8)
    block = block_module.Merge()
    hp = kerastuner.HyperParameters()

    block.build(hp, [ak.Input(shape=input_shape_1).build(),
                     ak.Input(shape=input_shape_2).build()])

    assert name_in_hps('merge_type', hp)


def test_temporal_reduction():
    input_shape = (32, 10)
    block = block_module.TemporalReduction()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('reduction_type', hp)


def test_spatial_reduction():
    input_shape = (32, 32, 3)
    block = block_module.SpatialReduction()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('reduction_type', hp)


def test_embedding_block():
    input_shape = (32,)
    block = block_module.EmbeddingBlock()
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert name_in_hps('pretraining', hp)
    assert name_in_hps('embedding_dim', hp)
