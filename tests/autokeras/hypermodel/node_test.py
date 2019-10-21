import copy

import numpy as np
import pytest
import tensorflow as tf

from autokeras.hypermodel import node
from tests import common


def test_structured_data_input_col_type_without_name():
    num_data = 500
    train_x = common.generate_structured_data(num_data)
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(
            column_types=common.COLUMN_TYPES_FROM_NUMPY)
        input_node.transform(train_x)
    assert str(info.value) == 'Column names must be specified.'


def test_structured_data_input_less_col_name():
    (x, _), _1 = common.dataframe_numpy()
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(
            column_names=common.LESS_COLUMN_NAMES_FROM_CSV)
        input_node.transform(x)
    assert 'Expect column_names to have length' in str(info.value)


def test_structured_data_input_name_type_mismatch():
    (x, _), _1 = common.dataframe_dataframe()
    column_types = copy.copy(common.COLUMN_TYPES_FROM_CSV)
    column_types['age_'] = column_types.pop('age')
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(column_types=column_types)
        input_node.transform(x)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


def test_structured_data_input_unsupported_type():
    x = 'unknown'
    with pytest.raises(TypeError) as info:
        input_node = node.StructuredDataInput(
            column_names=common.COLUMN_TYPES_FROM_NUMPY,
            column_types=common.COLUMN_TYPES_FROM_NUMPY)
        input_node.transform(x)
    assert 'Unsupported type' in str(info.value)


def test_structured_data_input_transform():
    (x, _), _1 = common.dataframe_dataframe()
    input_node = node.StructuredDataInput()
    input_node.transform(x)
    assert input_node.column_names[0] == 'sex'
    assert input_node.column_types == common.COLUMN_TYPES_FROM_CSV


def test_partial_column_types():
    input_node = node.StructuredDataInput(
        column_names=common.COLUMN_NAMES_FROM_CSV,
        column_types=common.PARTIAL_COLUMN_TYPES_FROM_CSV)
    (x, y), (val_x, val_y) = common.dataframe_numpy()
    dataset = x.values.astype(np.unicode)
    input_node.transform(dataset)
    assert input_node.column_types['fare'] == 'categorical'


def test_image_input():
    x = common.generate_data()
    input_node = node.ImageInput()
    assert isinstance(input_node.transform(x), tf.data.Dataset)


def test_image_input_with_three_dim():
    x = common.generate_data(shape=(32, 32))
    input_node = node.ImageInput()
    x = input_node.transform(x)
    assert isinstance(x, tf.data.Dataset)
    for a in x:
        assert a.shape == (32, 32, 1)
        break


def test_image_input_with_illegal_dim():
    x = common.generate_data(shape=(32,))
    input_node = node.ImageInput()
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to have 3' in str(info.value)


def test_image_input_unsupported_type():
    x = 'unknown'
    input_node = node.ImageInput()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to be numpy' in str(info.value)


def test_image_input_numerical():
    x = np.array([[['unknown']]])
    input_node = node.ImageInput()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to ImageInput to be numerical' in str(info.value)


def test_input_type_error():
    x = 'unknown'
    input_node = node.Input()
    with pytest.raises(TypeError) as info:
        input_node._check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numpy' in str(info.value)


def test_input_numerical():
    x = np.array([[['unknown']]])
    input_node = node.Input()
    with pytest.raises(TypeError) as info:
        input_node._check(x)
        x = input_node.transform(x)
    assert 'Expect the data to Input to be numerical' in str(info.value)


def test_text_input_type_error():
    x = 'unknown'
    input_node = node.TextInput()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to be numpy' in str(info.value)


def test_text_input_with_illegal_dim():
    x = common.generate_data(shape=(32,))
    input_node = node.TextInput()
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to have 1' in str(info.value)


def test_text_string():
    x = np.array([1, 2])
    input_node = node.TextInput()
    with pytest.raises(TypeError) as info:
        x = input_node.transform(x)
    assert 'Expect the data to TextInput to be strings' in str(info.value)
