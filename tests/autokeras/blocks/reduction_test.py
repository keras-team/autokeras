import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from tests import utils


def test_merge_build_return_tensor():
    block = blocks.Merge()

    outputs = block.build(
        kerastuner.HyperParameters(),
        [tf.keras.Input(shape=(32,), dtype=tf.float32),
         tf.keras.Input(shape=(4, 8), dtype=tf.float32)])

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_merge_deserialize_to_merge():
    serialized_block = blocks.serialize(blocks.Merge())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.Merge)


def test_merge_get_config_has_all_attributes():
    block = blocks.Merge()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.Merge.__init__).issubset(config.keys())


def test_temporal_build_return_tensor():
    block = blocks.TemporalReduction()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 10), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_temporal_deserialize_to_temporal():
    serialized_block = blocks.serialize(blocks.TemporalReduction())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TemporalReduction)


def test_temporal_get_config_has_all_attributes():
    block = blocks.TemporalReduction()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.TemporalReduction.__init__).issubset(config.keys())


def test_spatial_build_return_tensor():
    block = blocks.SpatialReduction()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_spatial_deserialize_to_spatial():
    serialized_block = blocks.serialize(blocks.SpatialReduction())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.SpatialReduction)


def test_spatial_get_config_has_all_attributes():
    block = blocks.SpatialReduction()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.SpatialReduction.__init__).issubset(config.keys())
