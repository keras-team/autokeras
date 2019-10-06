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
        input_node.fit(train_x)
    assert str(info.value) == 'Column names must be specified.'


def test_structured_data_input_less_col_name():
    (x, _), _1 = common.dataframe_numpy()
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(
            column_names=common.LESS_COLUMN_NAMES_FROM_CSV)
        input_node.fit(x)
    assert 'Expect column_names to have length' in str(info.value)


def test_structured_data_input_name_type_mismatch():
    (x, _), _1 = common.dataframe_dataframe()
    column_types = copy.copy(common.COLUMN_TYPES_FROM_CSV)
    column_types['age_'] = column_types.pop('age')
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(column_types=column_types)
        input_node.fit(x)
    assert 'Column_names and column_types are mismatched.' in str(info.value)


def test_structured_data_input_unsupported_type():
    x = 'unknown'
    with pytest.raises(TypeError) as info:
        input_node = node.StructuredDataInput(
            column_names=common.COLUMN_TYPES_FROM_NUMPY,
            column_types=common.COLUMN_TYPES_FROM_NUMPY)
        input_node.fit(x)
    assert 'Unsupported type' in str(info.value)


def test_structured_data_input_transform():
    (x, _), _1 = common.dataframe_dataframe()
    input_node = node.StructuredDataInput()
    input_node.fit(x)
    input_node.transform(x)
    assert input_node.column_names[0] == 'sex'
    assert input_node.column_types == common.COLUMN_TYPES_FROM_CSV


def test_partial_column_types():
    input_node = node.StructuredDataInput(
        column_names=common.COLUMN_NAMES_FROM_CSV,
        column_types=common.PARTIAL_COLUMN_TYPES_FROM_CSV)
    (x, y), (val_x, val_y) = common.dataframe_numpy()
    dataset = x.values.astype(np.unicode)
    input_node.fit(dataset)
    input_node.transform(dataset)
    assert input_node.column_types['fare'] == 'categorical'


def test_image_input():
    x = common.generate_data()
    input_node = node.ImageInput()
    input_node.fit(x)
    assert isinstance(input_node.transform(x), tf.data.Dataset)


def test_image_input_with_three_dim():
    x = common.generate_data(shape=(32, 32))
    input_node = node.ImageInput()
    input_node.fit(x)
    x = input_node.transform(x)
    assert isinstance(x, tf.data.Dataset)
    for a in x:
        assert a.shape == (32, 32, 1)
        break


def test_image_input_with_illegal_dim():
    x = common.generate_data(shape=(32,))
    input_node = node.ImageInput()
    input_node.fit(x)
    with pytest.raises(ValueError) as info:
        x = input_node.transform(x)
    assert 'Expect image input to have' in str(info.value)
