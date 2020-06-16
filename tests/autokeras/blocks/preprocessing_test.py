import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from tests import utils


def test_augment_build_return_tensor():
    block = blocks.ImageAugmentation()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_augment_deserialize_to_augment():
    serialized_block = blocks.serialize(blocks.ImageAugmentation())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.ImageAugmentation)


def test_augment_get_config_has_all_attributes():
    block = blocks.ImageAugmentation()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.ImageAugmentation.__init__).issubset(config.keys())


def test_ngram_build_return_tensor():
    block = blocks.TextToNgramVector()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(1,), dtype=tf.string))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_ngram_deserialize_to_ngram():
    serialized_block = blocks.serialize(blocks.TextToNgramVector())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TextToNgramVector)


def test_ngram_get_config_has_all_attributes():
    block = blocks.TextToNgramVector()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.TextToNgramVector.__init__).issubset(config.keys())


def test_int_seq_build_return_tensor():
    block = blocks.TextToIntSequence()

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(1,), dtype=tf.string))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_int_seq_deserialize_to_int_seq():
    serialized_block = blocks.serialize(blocks.TextToIntSequence())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.TextToIntSequence)


def test_int_seq_get_config_has_all_attributes():
    block = blocks.TextToIntSequence()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.TextToIntSequence.__init__).issubset(config.keys())


def test_cat_to_num_build_return_tensor():
    block = blocks.CategoricalToNumerical()
    block.column_names = ['a']
    block.column_types = {'a': 'num'}

    outputs = block.build(
        kerastuner.HyperParameters(),
        tf.keras.Input(shape=(1,), dtype=tf.string))

    assert len(nest.flatten(outputs)) == 1
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_cat_to_num_deserialize_to_cat_to_num():
    serialized_block = blocks.serialize(blocks.CategoricalToNumerical())

    block = blocks.deserialize(serialized_block)

    assert isinstance(block, blocks.CategoricalToNumerical)


def test_cat_to_num_get_config_has_all_attributes():
    block = blocks.CategoricalToNumerical()

    config = block.get_config()

    assert utils.get_func_args(
        blocks.CategoricalToNumerical.__init__).issubset(config.keys())
