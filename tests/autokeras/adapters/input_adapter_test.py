import copy

import numpy as np
import pytest
import tensorflow as tf

from autokeras.adapters import input_adapter
from tests import utils


def test_structured_data_input_col_type_without_name():
    num_data = 500
    train_x = utils.generate_structured_data(num_data)
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.StructuredDataInputAdapter(
            column_types=utils.COLUMN_TYPES_FROM_NUMPY)
        input_node.transform(train_x)
    assert str(info.value) == 'Column names must be specified.'


def test_structured_data_input_less_col_name():
    (x, _), _1 = utils.dataframe_numpy()
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.StructuredDataInputAdapter(
            column_names=utils.LESS_COLUMN_NAMES_FROM_CSV)
        input_node.transform(x)
    assert 'Expect column_names to have length' in str(info.value)


def test_structured_data_input_name_type_mismatch():
    (x, _), _1 = utils.dataframe_dataframe()
    column_types = copy.copy(utils.COLUMN_TYPES_FROM_CSV)
    column_types['age_'] = column_types.pop('age')
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.StructuredDataInputAdapter(
            column_types=column_types)
        input_node.transform(x)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


def test_structured_data_input_unsupported_type():
    x = 'unknown'
    with pytest.raises(TypeError) as info:
        input_node = input_adapter.StructuredDataInputAdapter(
            column_names=utils.COLUMN_TYPES_FROM_NUMPY,
            column_types=utils.COLUMN_TYPES_FROM_NUMPY)
        input_node.transform(x)
    assert 'Unsupported type' in str(info.value)


def test_structured_data_input_transform():
    (x, _), _1 = utils.dataframe_dataframe()
    input_node = input_adapter.StructuredDataInputAdapter()
    input_node.fit_transform(x)
    assert input_node.column_names[0] == 'sex'
    assert input_node.column_types == utils.COLUMN_TYPES_FROM_CSV


def test_partial_column_types():
    input_node = input_adapter.StructuredDataInputAdapter(
        column_names=utils.COLUMN_NAMES_FROM_CSV,
        column_types=utils.PARTIAL_COLUMN_TYPES_FROM_CSV)
    (x, y), (val_x, val_y) = utils.dataframe_numpy()
    dataset = x.values.astype(np.unicode)
    input_node.transform(dataset)
    assert input_node.column_types['fare'] == 'categorical'


def test_image_input():
    x = utils.generate_data()
    input_node = input_adapter.ImageInputAdapter()
    assert isinstance(input_node.transform(x), tf.data.Dataset)


def test_image_input_with_three_dim():
    x = utils.generate_data(shape=(32, 32))
    input_node = input_adapter.ImageInputAdapter()
    x = input_node.transform(x)
    assert isinstance(x, tf.data.Dataset)
    for a in x:
        assert a.shape == (32, 32, 1)
        break


def test_image_input_with_illegal_dim():
    x = utils.generate_data(shape=(32,))
    input_node = input_adapter.ImageInputAdapter()
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to have 3' in str(info.value)


def test_image_input_unsupported_type():
    x = 'unknown'
    input_node = input_adapter.ImageInputAdapter()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to be numpy' in str(info.value)


def test_image_input_numerical():
    x = np.array([[['unknown']]])
    input_node = input_adapter.ImageInputAdapter()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to be numerical' in str(info.value)


def test_input_type_error():
    x = 'unknown'
    input_node = input_adapter.InputAdapter()
    with pytest.raises(TypeError) as info:
        input_node.check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numpy' in str(info.value)


def test_input_numerical():
    x = np.array([[['unknown']]])
    input_node = input_adapter.InputAdapter()
    with pytest.raises(TypeError) as info:
        input_node.check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numerical' in str(info.value)


def test_text_input_type_error():
    x = 'unknown'
    input_node = input_adapter.TextInputAdapter()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to be numpy' in str(info.value)


def test_text_input_with_illegal_dim():
    x = utils.generate_data(shape=(32,))
    input_node = input_adapter.TextInputAdapter()
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to have 1' in str(info.value)


def test_text_string():
    x = np.array([1, 2])
    input_node = input_adapter.TextInputAdapter()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to be strings' in str(info.value)


def test_time_series_input_type_error():
    x = 'unknown'
    input_node = input_adapter.TimeseriesInputAdapter(2)
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data in TimeseriesInput to be numpy' in str(info.value)


def test_time_series_input_with_illegal_dim():
    x = utils.generate_data(shape=(32, 32))
    input_node = input_adapter.TimeseriesInputAdapter(2)
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect the data in TimeseriesInput to have 2' in str(info.value)


def test_time_series_input_col_type_without_name():
    num_data = 500
    train_x = utils.generate_structured_data(num_data)
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.TimeseriesInputAdapter(
            lookback=2,
            column_types=utils.COLUMN_TYPES_FROM_NUMPY)
        input_node.transform(train_x)
    assert str(info.value) == 'Column names must be specified.'


def test_time_series_input_less_col_name():
    (x, _), _1 = utils.dataframe_numpy()
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.TimeseriesInputAdapter(
            lookback=2,
            column_names=utils.LESS_COLUMN_NAMES_FROM_CSV)
        input_node.transform(x)
    assert 'Expect column_names to have length' in str(info.value)


def test_time_series_input_name_type_mismatch():
    (x, _), _1 = utils.dataframe_dataframe()
    column_types = copy.copy(utils.COLUMN_TYPES_FROM_CSV)
    column_types['age_'] = column_types.pop('age')
    with pytest.raises(ValueError) as info:
        input_node = input_adapter.TimeseriesInputAdapter(
            lookback=2,
            column_types=column_types)
        input_node.transform(x)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


def test_time_series_input_transform():
    x = utils.generate_data(shape=(32,))
    input_node = input_adapter.TimeseriesInputAdapter(2)
    x = input_node.transform(x)
    for row in x.as_numpy_iterator():
        assert row.ndim == 2
