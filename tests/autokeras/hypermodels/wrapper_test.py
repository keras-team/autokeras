import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras.hypermodels import wrapper
from tests.autokeras.hypermodels import utils


def test_image_block():
    utils.block_basic_exam(
        wrapper.ImageBlock(normalize=None, augment=None),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        [
            'block_type',
            'normalize',
            'augment',
        ],
    )


def test_text_block():
    utils.block_basic_exam(
        wrapper.TextBlock(),
        tf.keras.Input(shape=(1,), dtype=tf.string),
        ['vectorizer'],
    )


def test_structured_data_block():
    block = wrapper.StructuredDataBlock()
    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.NUMERICAL,
        '1': adapters.NUMERICAL,
    }
    outputs = utils.block_basic_exam(
        block,
        tf.keras.Input(shape=(2,), dtype=tf.string),
        [],
    )
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)


def test_timeseries_block():
    block = wrapper.TimeseriesBlock()
    block.column_names = ['0', '1']
    block.column_types = {
        '0': adapters.NUMERICAL,
        '1': adapters.NUMERICAL,
    }
    outputs = utils.block_basic_exam(
        block,
        tf.keras.Input(shape=(32, 2), dtype=tf.float32),
        [],
    )
    assert isinstance(nest.flatten(outputs)[0], tf.Tensor)
