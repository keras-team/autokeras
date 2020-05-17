import tensorflow as tf

from autokeras.hypermodels import preprocessing
from tests.autokeras.hypermodels import utils


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
