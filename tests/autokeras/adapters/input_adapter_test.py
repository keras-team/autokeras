# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from autokeras.adapters import input_adapter
from autokeras.utils import data_utils
from tests import utils


def test_structured_data_col_type_no_name_error():
    with pytest.raises(ValueError) as info:
        adapter = input_adapter.StructuredDataInputAdapter(
            column_types=utils.COLUMN_TYPES
        )
        adapter.transform(
            pd.read_csv(utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)
        )

    assert str(info.value) == "Column names must be specified."


def test_structured_data_input_less_col_name_error():
    with pytest.raises(ValueError) as info:
        adapter = input_adapter.StructuredDataInputAdapter(
            column_names=utils.COLUMN_NAMES[:-2]
        )
        adapter.fit_transform(pd.read_csv(utils.TRAIN_CSV_PATH))

    assert "Expect column_names to have length" in str(info.value)


def test_structured_data_input_name_type_mismatch_error():
    column_types = copy.copy(utils.COLUMN_TYPES)
    column_types["age_"] = column_types.pop("age")

    with pytest.raises(ValueError) as info:
        adapter = input_adapter.StructuredDataInputAdapter(column_types=column_types)
        adapter.transform(pd.read_csv(utils.TRAIN_CSV_PATH))

    assert "Column_names and column_types are mismatched." in str(info.value)


def test_structured_data_input_unsupported_type_error():
    with pytest.raises(TypeError) as info:
        adapter = input_adapter.StructuredDataInputAdapter(
            column_names=utils.COLUMN_NAMES,
            column_types=utils.COLUMN_TYPES,
        )
        adapter.transform("unknown")

    assert "Unsupported type" in str(info.value)


def test_structured_data_infer_col_types():
    adapter = input_adapter.StructuredDataInputAdapter()
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")

    adapter.fit_transform(x)

    assert adapter.column_types == utils.COLUMN_TYPES


def test_structured_data_restore_col_types_from_config():
    adapter = input_adapter.StructuredDataInputAdapter()
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")

    adapter.fit_transform(x)
    adapter = input_adapter.StructuredDataInputAdapter.from_config(
        adapter.get_config()
    )

    assert adapter.column_types == utils.COLUMN_TYPES


def test_structured_data_get_col_names():
    adapter = input_adapter.StructuredDataInputAdapter()
    x = pd.read_csv(utils.TRAIN_CSV_PATH)
    x.pop("survived")

    adapter.fit_transform(x)

    assert adapter.column_names[0] == "sex"


def test_structured_data_input_transform_to_dataset():
    x = tf.data.Dataset.from_tensor_slices(
        pd.read_csv(utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)
    )
    adapter = input_adapter.StructuredDataInputAdapter()

    x = adapter.fit_transform(x)

    assert isinstance(x, tf.data.Dataset)


def test_dont_infer_specified_column_types():
    column_types = copy.copy(utils.COLUMN_TYPES)
    column_types.pop("sex")
    column_types["age"] = "categorical"

    adapter = input_adapter.StructuredDataInputAdapter(
        column_names=utils.COLUMN_NAMES,
        column_types=column_types,
    )
    dataset = pd.read_csv(utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)
    adapter.transform(dataset)

    assert adapter.column_types["age"] == "categorical"


def test_image_input_adapter_transform_to_dataset():
    x = utils.generate_data()
    adapter = input_adapter.ImageInputAdapter()
    assert isinstance(adapter.transform(x), tf.data.Dataset)


def test_image_input_adapter_shape_is_list():
    x = utils.generate_data()
    adapter = input_adapter.ImageInputAdapter()
    adapter.fit_transform(x)
    assert isinstance(adapter.shape, list)
    assert all(map(lambda x: isinstance(x, int), adapter.shape))


def test_image_input_with_three_dim():
    x = utils.generate_data(shape=(32, 32))
    adapter = input_adapter.ImageInputAdapter()
    x = adapter.transform(x)
    assert isinstance(x, tf.data.Dataset)
    for a in x:
        assert a.shape[1:] == (32, 32, 1)
        break


def test_image_input_with_illegal_dim():
    x = utils.generate_data(shape=(32,))
    adapter = input_adapter.ImageInputAdapter()
    with pytest.raises(ValueError) as info:
        x = adapter.transform(x)
    assert "Expect the data to ImageInput to have 3" in str(info.value)


def test_image_input_unsupported_type():
    x = "unknown"
    adapter = input_adapter.ImageInputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.transform(x)
    assert "Expect the data to ImageInput to be numpy" in str(info.value)


def test_image_input_numerical():
    x = np.array([[["unknown"]]])
    adapter = input_adapter.ImageInputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.transform(x)
    assert "Expect the data to ImageInput to be numerical" in str(info.value)


def test_input_type_error():
    x = "unknown"
    adapter = input_adapter.InputAdapter()
    with pytest.raises(TypeError) as info:
        adapter.check(x)
        x = adapter.transform(x)
    assert "Expect the data to Input to be numpy" in str(info.value)


def test_input_numerical():
    x = np.array([[["unknown"]]])
    adapter = input_adapter.InputAdapter()
    with pytest.raises(TypeError) as info:
        adapter.check(x)
        x = adapter.transform(x)
    assert "Expect the data to Input to be numerical" in str(info.value)


def test_text_dataset():
    x = tf.data.Dataset.from_tensor_slices(np.array(["a b c", "b b c"]))
    adapter = input_adapter.TextInputAdapter()
    x = adapter.transform(x)
    assert data_utils.dataset_shape(x).as_list() == [None, 1]
    assert isinstance(x, tf.data.Dataset)


def test_text_dataset_batch():
    x = tf.data.Dataset.from_tensor_slices(np.array(["a b c", "b b c"])).batch(32)
    adapter = input_adapter.TextInputAdapter()
    x = adapter.transform(x)
    assert data_utils.dataset_shape(x).as_list() == [None, 1]
    assert isinstance(x, tf.data.Dataset)


def test_text_np():
    x = np.array(["a b c", "b b c"])
    adapter = input_adapter.TextInputAdapter()
    x = adapter.transform(x)
    assert data_utils.dataset_shape(x).as_list() == [None, 1]
    assert isinstance(x, tf.data.Dataset)


def test_text_input_type_error():
    x = "unknown"
    adapter = input_adapter.TextInputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.transform(x)
    assert "Expect the data to TextInput to be numpy" in str(info.value)


def test_text_input_with_illegal_dim():
    x = utils.generate_data(shape=(32,))
    adapter = input_adapter.TextInputAdapter()
    with pytest.raises(ValueError) as info:
        x = adapter.transform(x)
    assert "Expect the data to TextInput to have 1" in str(info.value)


def test_text_string():
    x = np.array([1, 2])
    adapter = input_adapter.TextInputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.transform(x)
    assert "Expect the data to TextInput to be strings" in str(info.value)


def test_time_series_input_type_error():
    x = "unknown"
    adapter = input_adapter.TimeseriesInputAdapter(2)
    with pytest.raises(TypeError) as info:
        x = adapter.transform(x)
    assert "Expect the data in TimeseriesInput to be numpy" in str(info.value)


def test_time_series_input_with_illegal_dim():
    x = utils.generate_data(shape=(32, 32))
    adapter = input_adapter.TimeseriesInputAdapter(2)
    with pytest.raises(ValueError) as info:
        x = adapter.transform(x)
    assert "Expect the data in TimeseriesInput to have 2" in str(info.value)


def test_time_series_input_col_type_without_name():
    train_x = pd.read_csv(utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)
    with pytest.raises(ValueError) as info:
        adapter = input_adapter.TimeseriesInputAdapter(
            lookback=2, column_types=utils.COLUMN_TYPES
        )
        adapter.transform(train_x)
    assert str(info.value) == "Column names must be specified."


def test_time_series_input_less_col_name():
    with pytest.raises(ValueError) as info:
        adapter = input_adapter.TimeseriesInputAdapter(
            lookback=2, column_names=utils.COLUMN_NAMES[:-2]
        )
        adapter.transform(pd.read_csv(utils.TRAIN_CSV_PATH))
    assert "Expect column_names to have length" in str(info.value)


def test_time_series_input_name_type_mismatch():
    column_types = copy.copy(utils.COLUMN_TYPES)
    column_types["age_"] = column_types.pop("age")
    with pytest.raises(ValueError) as info:
        adapter = input_adapter.TimeseriesInputAdapter(
            lookback=2, column_types=column_types
        )
        adapter.transform(pd.read_csv(utils.TRAIN_CSV_PATH))
    assert "Column_names and column_types are mismatched." in str(info.value)


def test_time_series_input_restore_look_back():
    adapter = input_adapter.TimeseriesInputAdapter(2)

    adapter = input_adapter.TimeseriesInputAdapter.from_config(adapter.get_config())

    assert adapter.lookback == 2


def test_time_series_input_transform_df_to_dataset():
    adapter = input_adapter.TimeseriesInputAdapter(2)

    x = adapter.fit_transform(pd.DataFrame(utils.generate_data(shape=(32,))))

    assert isinstance(x, tf.data.Dataset)


def test_time_series_input_transform():
    x = utils.generate_data(shape=(32,))
    adapter = input_adapter.TimeseriesInputAdapter(2)
    x = adapter.transform(x)
    for row in x.as_numpy_iterator():
        assert row.ndim == 3
