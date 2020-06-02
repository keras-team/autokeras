import numpy as np
import pytest
import tensorflow as tf

from autokeras.adapters import output_adapter
from tests import utils


def test_y_is_pd_series():
    (x, y), (val_x, val_y) = utils.dataframe_series()
    adapter = output_adapter.ClassificationHeadAdapter(name='a')
    adapter.fit_transform(y)
    assert isinstance(adapter.transform(y), tf.data.Dataset)


def test_unsupported_types():
    y = 1
    adapter = output_adapter.ClassificationHeadAdapter(name='a')
    with pytest.raises(TypeError) as info:
        adapter.check(y)
    assert 'Expect the target data' in str(info.value)


def test_one_class():
    y = np.array(['a', 'a', 'a'])
    adapter = output_adapter.ClassificationHeadAdapter(name='a')
    with pytest.raises(ValueError) as info:
        adapter.fit_before_convert(y)
    assert 'Expect the target data' in str(info.value)


def test_infer_num_classes():
    y = utils.generate_one_hot_labels(dtype='dataset')
    adapter = output_adapter.ClassificationHeadAdapter(name='a')
    y = adapter.fit(y)
    assert adapter.num_classes == 10


def test_infer_two_classes():
    y = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32)
    adapter = output_adapter.ClassificationHeadAdapter(name='a')
    y = adapter.fit(y)
    assert adapter.num_classes == 2


def test_check_data_shape():
    y = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 5)).batch(32)
    adapter = output_adapter.ClassificationHeadAdapter(name='a', num_classes=5)
    adapter.fit(y)


def test_check_data_shape_two_classes():
    y = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32)
    adapter = output_adapter.ClassificationHeadAdapter(name='a', num_classes=2)
    adapter.fit(y)


def test_check_data_shape_error():
    y = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 3)).batch(32)
    adapter = output_adapter.ClassificationHeadAdapter(name='a', num_classes=5)
    with pytest.raises(ValueError) as info:
        adapter.fit(y)
    assert 'Expect the target data for a to have shape' in str(info.value)


def test_multi_label_two_classes():
    y = np.random.rand(10, 2)
    adapter = output_adapter.ClassificationHeadAdapter(name='a', multi_label=True)
    adapter.fit_transform(y)
    assert adapter.label_encoder is None


def test_multi_label_postprocessing():
    y = np.random.rand(10, 3)
    adapter = output_adapter.ClassificationHeadAdapter(name='a', multi_label=True)
    adapter.fit_transform(y)
    y = adapter.postprocess(y)
    assert set(y.flatten().tolist()) == set([1, 0])
