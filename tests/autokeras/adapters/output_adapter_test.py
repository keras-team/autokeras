import numpy as np
import pytest
import tensorflow as tf

from autokeras.adapters import output_adapter
from tests import utils


def test_y_is_pd_series():
    (x, y), (val_x, val_y) = utils.dataframe_series()
    head = output_adapter.ClassificationHeadAdapter(name='a')
    head.fit_transform(y)
    assert isinstance(head.transform(y), tf.data.Dataset)


def test_unsupported_types():
    y = 1
    head = output_adapter.ClassificationHeadAdapter(name='a')
    with pytest.raises(TypeError) as info:
        head.fit_transform(y)
    assert 'Expect the target data' in str(info.value)


def test_one_class():
    y = np.array(['a', 'a', 'a'])
    head = output_adapter.ClassificationHeadAdapter(name='a')
    with pytest.raises(ValueError) as info:
        head.fit_transform(y)
    assert 'Expect the target data' in str(info.value)


def test_tf_dataset():
    y = utils.generate_one_hot_labels(dtype='dataset')
    head = output_adapter.ClassificationHeadAdapter(name='a')
    y = head.fit_transform(y)
    assert isinstance(head.transform(y), tf.data.Dataset)
