import tensorflow as tf

from autokeras.blocks import preprocessing
from tests.autokeras.blocks import utils


def test_image_augmentation():
    utils.block_basic_exam(
        preprocessing.ImageAugmentation(),
        tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32),
        ['vertical_flip', 'horizontal_flip'],
    )


def test_text_to_ngram_vector():
    utils.block_basic_exam(
        preprocessing.TextToNgramVector(),
        tf.keras.Input(shape=(1,), dtype=tf.string),
        ['ngrams'],
    )


def test_text_to_int_sequence():
    utils.block_basic_exam(
        preprocessing.TextToIntSequence(),
        tf.keras.Input(shape=(1,), dtype=tf.string),
        ['output_sequence_length'],
    )


def test_categorical_to_numerical():
    block = preprocessing.CategoricalToNumerical()
    block.column_names = ['a']
    block.column_types = {'a': 'num'}
    utils.block_basic_exam(
        block,
        tf.keras.Input(shape=(1,), dtype=tf.string),
        [],
    )
