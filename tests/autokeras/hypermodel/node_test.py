import pytest

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
    with pytest.raises(ValueError) as info:
        input_node = node.StructuredDataInput(
            column_types=common.COLUMN_TYPES_FROM_CSV)
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
