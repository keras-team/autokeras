import functools

import kerastuner
import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from autokeras.hypermodel import block
from autokeras.hypermodel import head
from autokeras.hypermodel import preprocessor
from tests import common


def map_func(x, instance=None, dtype=tf.float32):
    return tf.py_function(instance.transform,
                          inp=[x],
                          Tout=(dtype,))


def run_preprocessor(instance, x, y=None, dtype=tf.float32):
    dataset = tf.data.Dataset.zip((x, y))
    instance.set_hp(kerastuner.HyperParameters())
    for temp_x, temp_y in dataset:
        instance.update(temp_x, temp_y)
    instance.finalize()
    instance.set_config(instance.get_config())

    weights = instance.get_weights()
    instance.clear_weights()
    instance.set_weights(weights)

    for temp_x, _ in dataset:
        instance.transform(temp_x, True)

    new_dataset = x.map(functools.partial(map_func,
                                          instance=instance,
                                          dtype=dtype))
    for _ in new_dataset:
        pass
    return new_dataset


def test_normalize():
    dataset = common.generate_data(dtype='dataset')
    new_dataset = run_preprocessor(preprocessor.Normalization(),
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   dtype=tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_sequence():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = run_preprocessor(preprocessor.TextToIntSequence(),
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.int64)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_ngram():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    new_dataset = run_preprocessor(preprocessor.TextToNgramVector(),
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_augment():
    dataset = common.generate_data(dtype='dataset')
    new_dataset = run_preprocessor(preprocessor.ImageAugmentation(seed=common.SEED),
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering():
    dataset = common.generate_structured_data(dtype='dataset')
    feature = preprocessor.FeatureEngineering()
    feature.input_node = ak.StructuredDataInput(
        column_names=common.COLUMN_NAMES_FROM_NUMPY,
        column_types=common.COLUMN_TYPES_FROM_NUMPY)
    new_dataset = run_preprocessor(feature,
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_feature_engineering_fix_keyerror():
    dataset = common.generate_structured_data(num_instances=100, dtype='dataset')
    feature = preprocessor.FeatureEngineering()
    feature.input_node = ak.StructuredDataInput(
        column_names=common.COLUMN_NAMES_FROM_NUMPY,
        column_types=common.COLUMN_TYPES_FROM_NUMPY)
    new_dataset = run_preprocessor(feature,
                                   dataset,
                                   common.generate_data(dtype='dataset'),
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_classifier():
    dataset = common.generate_data(11, (32,), dtype='dataset')
    y = common.generate_one_hot_labels(11, dtype='dataset')
    instance = preprocessor.LightGBMBlock()
    instance.lightgbm_block = preprocessor.LightGBMClassifier()
    new_dataset = run_preprocessor(instance,
                                   dataset,
                                   y,
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)


def test_lgbm_regressor():
    dataset = common.generate_data(11, (32,), dtype='dataset')
    y = common.generate_data(11, (1,), dtype='dataset')
    instance = preprocessor.LightGBMBlock()
    instance.lightgbm_block = preprocessor.LightGBMRegressor()
    new_dataset = run_preprocessor(instance,
                                   dataset,
                                   y,
                                   tf.float32)
    assert isinstance(new_dataset, tf.data.Dataset)
