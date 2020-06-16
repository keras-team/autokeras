import kerastuner
import pytest
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from autokeras.blocks import basic
from tests import utils


def test_resnet_build_return_tensor():
    block = blocks.ResNetBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_resnet_deserialize_to_resnet():
    serialized_block = blocks.serialize(blocks.ResNetBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ResNetBlock)


def test_resnet_get_config_has_all_attributes():
    block = blocks.ResNetBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.ResNetBlock.__init__).issubset(config.keys())


def test_resnet_init_error_with_include_top():
    with pytest.raises(ValueError) as info:
        blocks.ResNetBlock(include_top=True)

    assert 'Argument "include_top" is not' in str(info.value)


def test_resnet_init_error_with_input_shape():
    with pytest.raises(ValueError) as info:
        blocks.ResNetBlock(input_shape=(10,))

    assert 'Argument "input_shape" is not' in str(info.value)


def test_xception_build_return_tensor():
    block = blocks.XceptionBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_xception_deserialize_to_xception():
    serialized_block = blocks.serialize(blocks.XceptionBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.XceptionBlock)


def test_xception_get_config_has_all_attributes():
    block = blocks.XceptionBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.XceptionBlock.__init__).issubset(config.keys())


def test_xception_init_error_with_include_top():
    with pytest.raises(ValueError) as info:
        basic.XceptionBlock(include_top=True)

    assert 'Argument "include_top" is not' in str(info.value)


def test_xception_init_error_with_input_shape():
    with pytest.raises(ValueError) as info:
        basic.XceptionBlock(input_shape=(10,))

    assert 'Argument "input_shape" is not' in str(info.value)


def test_conv_build_return_tensor():
    block = blocks.ConvBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_conv_deserialize_to_conv():
    serialized_block = blocks.serialize(blocks.ConvBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ConvBlock)


def test_conv_get_config_has_all_attributes():
    block = blocks.ConvBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.ConvBlock.__init__).issubset(config.keys())


def test_rnn_build_return_tensor():
    block = blocks.RNNBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_rnn_deserialize_to_rnn():
    serialized_block = blocks.serialize(blocks.RNNBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.RNNBlock)


def test_rnn_get_config_has_all_attributes():
    block = blocks.RNNBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.RNNBlock.__init__).issubset(config.keys())


def test_dense_build_return_tensor():
    block = blocks.DenseBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32,), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_dense_deserialize_to_dense():
    serialized_block = blocks.serialize(blocks.DenseBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.DenseBlock)


def test_dense_get_config_has_all_attributes():
    block = blocks.DenseBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.DenseBlock.__init__).issubset(config.keys())


def test_embed_build_return_tensor():
    block = blocks.Embedding()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32,), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_embed_deserialize_to_embed():
    serialized_block = blocks.serialize(blocks.Embedding())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.Embedding)


def test_embed_get_config_has_all_attributes():
    block = blocks.Embedding()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.Embedding.__init__).issubset(config.keys())
