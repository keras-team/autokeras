import functools

import tensorflow as tf
import numpy as np
import kerastuner

from autokeras.hypermodel import processor


def test_normalize():
    normalize = processor.Normalize()
    x_train = np.random.rand(100, 32, 32, 3)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    hp = kerastuner.HyperParameters()
    normalize.fit(dataset, hp)
    # dataset.map(lambda x: x - normalize.mean)
    new_dataset = dataset.map(functools.partial(normalize.transform, hp=hp))
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = processor.TextToSequenceVector()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = tokenize.fit_transform(kerastuner.HyperParameters(), dataset)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = processor.TextToNgramVector()
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = tokenize.fit_transform(kerastuner.HyperParameters(), dataset)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram_with_labels():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenize = processor.TextToNgramVector(labels=[1, 0, 1])
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = tokenize.fit_transform(kerastuner.HyperParameters(), dataset)
    assert isinstance(new_dataset, tf.data.Dataset)
