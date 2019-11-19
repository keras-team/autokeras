import kerastuner
import numpy as np
import pytest
import tensorflow as tf

from autokeras.hypermodel import head as head_module
from autokeras.hypermodel import node as node_module
from tests import common


def test_y_is_pd_series():
    (x, y), (val_x, val_y) = common.dataframe_series()
    head = head_module.ClassificationHead()
    head.fit_transform(y)
    if not isinstance(head.transform(y), tf.data.Dataset):
        raise AssertionError()


def test_unsupported_types():
    y = 1
    head = head_module.ClassificationHead(name='a')
    with pytest.raises(TypeError) as info:
        head.fit_transform(y)
    if 'Expect the target data' not in str(info.value):
        raise AssertionError()


def test_one_class():
    y = np.array(['a', 'a', 'a'])
    head = head_module.ClassificationHead(name='a')
    with pytest.raises(ValueError) as info:
        head.fit_transform(y)
    if 'Expect the target data' not in str(info.value):
        raise AssertionError()


def test_two_classes():
    y = np.array(['a', 'a', 'a', 'b'])
    head = head_module.ClassificationHead(name='a')
    head.fit_transform(y)
    head.output_shape = (1,)
    head.build(kerastuner.HyperParameters(), node_module.Input(shape=(32,)).build())
    if head.loss != 'binary_crossentropy':
        raise AssertionError()


def test_three_classes():
    y = np.array(['a', 'a', 'c', 'b'])
    head = head_module.ClassificationHead(name='a')
    head.fit_transform(y)
    if head.loss != 'categorical_crossentropy':
        raise AssertionError()
