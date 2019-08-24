import functools
import kerastuner
import numpy as np
import tensorflow as tf

from autokeras.hypermodel import preprocessor


def test_normalize():
    normalize = preprocessor.Normalization()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    normalize.set_hp(kerastuner.HyperParameters())
    for x in dataset:
        normalize.update(x)
    normalize.finalize()
    normalize.set_config(normalize.get_config())

    weights = normalize.get_weights()
    normalize.clear_weights()
    normalize.set_weights(weights)

    for a in dataset:
        normalize.transform(a)

    def map_func(x):
        return tf.py_function(normalize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
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
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.int64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
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
    tokenize.finalize()
    tokenize.set_config(tokenize.get_config())

    weights = tokenize.get_weights()
    tokenize.clear_weights()
    tokenize.set_weights(weights)

    for a in dataset:
        tokenize.transform(a)

    def map_func(x):
        return tf.py_function(tokenize.transform,
                              inp=[x],
                              Tout=(tf.float64,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    raw_images = tf.random.normal([1000, 32, 32, 3], mean=-1, stddev=4)
    augment = preprocessor.ImageAugmentation(seed=5)
    dataset = tf.data.Dataset.from_tensor_slices(raw_images)
    hp = kerastuner.HyperParameters()
    augment.set_hp(hp)
    augment.set_config(augment.get_config())
    for a in dataset:
        augment.transform(a, True)

    def map_func(x):
        return tf.py_function(functools.partial(augment.transform, fit=True),
                              inp=[x],
                              Tout=(tf.float32,))

    new_dataset = dataset.map(map_func)
    for _ in new_dataset:
        pass
    assert isinstance(new_dataset, tf.data.Dataset)
