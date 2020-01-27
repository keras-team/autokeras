from unittest import mock

import kerastuner
import pytest
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodel import block as block_module
from tests import common


def test_type_error_for_call():
    block = block_module.ConvBlock()
    with pytest.raises(TypeError) as info:
        block(block)
    assert 'Expect the inputs to layer' in str(info.value)


@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.__init__')
@mock.patch('autokeras.hypermodel.block.resnet.HyperResNet.build')
def test_resnet_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.ResNetBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('version', hp)
    assert common.name_in_hps('pooling', hp)
    assert init.called
    assert build.called


@mock.patch('autokeras.hypermodel.block.xception.HyperXception.__init__')
@mock.patch('autokeras.hypermodel.block.xception.HyperXception.build')
def test_xception_block(init, build):
    input_shape = (32, 32, 3)
    block = block_module.XceptionBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('activation', hp)
    assert common.name_in_hps('initial_strides', hp)
    assert common.name_in_hps('num_residual_blocks', hp)
    assert common.name_in_hps('pooling', hp)
    assert init.called
    assert build.called


def test_conv_block():
    input_shape = (32, 32, 3)
    block = block_module.ConvBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('kernel_size', hp)
    assert common.name_in_hps('num_blocks', hp)
    assert common.name_in_hps('separable', hp)


def test_rnn_block():
    input_shape = (32, 10)
    block = block_module.RNNBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('bidirectional', hp)
    assert common.name_in_hps('layer_type', hp)
    assert common.name_in_hps('num_layers', hp)


def test_dense_block():
    input_shape = (32,)
    block = block_module.DenseBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('num_layers', hp)
    assert common.name_in_hps('use_batchnorm', hp)


def test_merge():
    input_shape_1 = (32,)
    input_shape_2 = (4, 8)
    block = block_module.Merge()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, [ak.Input(shape=input_shape_1).build(),
                     ak.Input(shape=input_shape_2).build()])

    assert common.name_in_hps('merge_type', hp)


def test_temporal_reduction():
    input_shape = (32, 10)
    block = block_module.TemporalReduction()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('reduction_type', hp)


def test_spatial_reduction():
    input_shape = (32, 32, 3)
    block = block_module.SpatialReduction()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('reduction_type', hp)


def test_embedding_block():
    input_shape = (32,)
    block = block_module.Embedding()
    block.max_features = 100
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.Input(shape=input_shape).build())

    assert common.name_in_hps('pretraining', hp)
    assert common.name_in_hps('embedding_dim', hp)


def test_image_block():
    block = block_module.ImageBlock(normalize=None, augment=None)
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.ImageInput(shape=(32, 32, 3)).build())

    assert common.name_in_hps('block_type', hp)
    assert common.name_in_hps('normalize', hp)
    assert common.name_in_hps('augment', hp)


def test_text_block():
    block = block_module.TextBlock()
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    block.build(hp, ak.TextInput(shape=(1,)).build())

    assert common.name_in_hps('vectorizer', hp)


def test_structured_data_block():
    block = block_module.StructuredDataBlock()
    block.num_heads = 1
    block.column_names = ['0', '1']
    block.column_types = {
        '0': ak.StructuredDataInput.CATEGORICAL,
        '1': ak.StructuredDataInput.CATEGORICAL,
    }
    block.set_state(block.get_state())
    hp = kerastuner.HyperParameters()

    output = block.build(hp, ak.StructuredDataInput(shape=(2,)).build())

    assert isinstance(output, tf.Tensor)
