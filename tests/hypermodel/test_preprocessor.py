import kerastuner
import numpy as np
import tensorflow as tf

from autokeras.hypermodel import preprocessor
from autokeras.hypermodel import block

def test_normalize():
    normalize = preprocessor.Normalization()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    normalize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        normalize.update(x)

    def map_func(x):
        return tf.py_function(normalize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToIntSequence()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.int64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        break
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = preprocessor.TextToNgramVector()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    tokenize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        tokenize.update(x)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    raw_images = tf.random.normal([1000, 32, 32, 3], mean=-1, stddev=4)
    augmenter = preprocessor.ImageAugmentation()
    dataset = tf.data.Dataset.from_tensor_slices(raw_images)
    hp = kerastuner.HyperParameters()
    block.set_hp_value(hp, 'whether_rotation_range', True)
    block.set_hp_value(hp, 'whether_random_crop', True)
    block.set_hp_value(hp, 'whether_brightness_range', True)
    block.set_hp_value(hp, 'whether_saturation_range', True)
    block.set_hp_value(hp, 'whether_contrast_range', True)
    augmenter.set_hp(hp)
    def map_func(x):
        return tf.py_function(augmenter.transform,
                              inp=[x],
                              Tout=(tf.float32,))

    new_dataset = dataset.map(map_func)
    assert isinstance(new_dataset, tf.data.Dataset)
