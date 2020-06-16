import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import blocks
from tests import utils


def test_image_build_return_tensor():
    block = blocks.ImageBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_image_deserialize_to_image():
    serialized_block = blocks.serialize(blocks.ImageBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ImageBlock)


def test_image_get_config_has_all_attributes():
    block = blocks.ImageBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.ImageBlock.__init__).issubset(config.keys())


def test_text_build_return_tensor():
    block = blocks.TextBlock()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(1,), dtype=tf.string))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_text_deserialize_to_text():
    serialized_block = blocks.serialize(blocks.TextBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TextBlock)


def test_text_get_config_has_all_attributes():
    block = blocks.TextBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.TextBlock.__init__).issubset(config.keys())


def test_structured_build_return_tensor():
    block = blocks.StructuredDataBlock()
    block.column_names = ['0', '1']
    block.column_types = {'0': adapters.NUMERICAL,
                          '1': adapters.NUMERICAL}

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(2,), dtype=tf.string))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_structured_deserialize_to_structured():
    serialized_block = blocks.serialize(blocks.StructuredDataBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.StructuredDataBlock)


def test_structured_get_config_has_all_attributes():
    block = blocks.StructuredDataBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.StructuredDataBlock.__init__).issubset(config.keys())


def test_timeseries_build_return_tensor():
    block = blocks.TimeseriesBlock()
    block.column_names = ['0', '1']
    block.column_types = {'0': adapters.NUMERICAL,
                          '1': adapters.NUMERICAL}

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 2), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_timeseries_deserialize_to_timeseries():
    serialized_block = blocks.serialize(blocks.TimeseriesBlock())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TimeseriesBlock)


def test_timeseries_get_config_has_all_attributes():
    block = blocks.TimeseriesBlock()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.TimeseriesBlock.__init__).issubset(config.keys())
