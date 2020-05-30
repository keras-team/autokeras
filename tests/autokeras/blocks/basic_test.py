from unittest import mock

import pytest
import tensorflow as tf

from autokeras.blocks import basic
from tests.autokeras.blocks import utils


def test_type_error_for_call():
    block = basic.ConvBlock()
    with pytest.raises(TypeError) as info:
        block(block)
    assert 'Expect the inputs to layer' in str(info.value)


@mock.patch('autokeras.blocks.basic.resnet.HyperResNet.__init__')
@mock.patch('autokeras.blocks.basic.resnet.HyperResNet.build')
def test_resnet_block(init, build):
    utils.block_basic_exam(
        basic.ResNetBlock(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        ['version', 'pooling'],
    )
    assert init.called
    assert build.called


@mock.patch('autokeras.blocks.basic.xception.HyperXception.__init__')
@mock.patch('autokeras.blocks.basic.xception.HyperXception.build')
def test_xception_block(init, build):
    utils.block_basic_exam(
        basic.XceptionBlock(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        [
            'activation',
            'initial_strides',
            'num_residual_blocks',
            'pooling',
        ])
    assert init.called
    assert build.called


def test_conv_block():
    utils.block_basic_exam(
        basic.ConvBlock(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        [
            'kernel_size',
            'num_blocks',
            'separable',
        ])


def test_rnn_block():
    utils.block_basic_exam(
        basic.RNNBlock(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
        [
            'bidirectional',
            'layer_type',
            'num_layers',
        ])


def test_rnn_attention_block():
    utils.block_basic_exam(
        basic.AttentionBlock(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32),
        []
    )


def test_dense_block():
    utils.block_basic_exam(
        basic.DenseBlock(),
        tf.keras.Input(shape=(32,), dtype=tf.float32),
        [
            'num_layers',
            'use_batchnorm',
        ])


def test_embedding_block():
    utils.block_basic_exam(
        basic.Embedding(),
        tf.keras.Input(shape=(32,), dtype=tf.float32),
        [
            'pretraining',
            'embedding_dim',
        ])
