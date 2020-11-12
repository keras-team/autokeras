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


import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from autokeras.adapters import input_adapters
from autokeras.utils import data_utils
from tests import utils


def test_structured_data_input_unsupported_type_error():
    with pytest.raises(TypeError) as info:
        adapter = input_adapters.StructuredDataAdapter()
        adapter.adapt("unknown", batch_size=32)

    assert "Unsupported type" in str(info.value)


def test_structured_data_input_transform_to_dataset():
    x = tf.data.Dataset.from_tensor_slices(
        pd.read_csv(utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)
    )
    adapter = input_adapters.StructuredDataAdapter()

    x = adapter.adapt(x, batch_size=32)

    assert isinstance(x, tf.data.Dataset)


def test_image_input_adapter_transform_to_dataset():
    x = utils.generate_data()
    adapter = input_adapters.ImageAdapter()
    assert isinstance(adapter.adapt(x, batch_size=32), tf.data.Dataset)


def test_image_input_unsupported_type():
    x = "unknown"
    adapter = input_adapters.ImageAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data to ImageInput to be numpy" in str(info.value)


def test_image_input_numerical():
    x = np.array([[["unknown"]]])
    adapter = input_adapters.ImageAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data to ImageInput to be numerical" in str(info.value)


def test_input_type_error():
    x = "unknown"
    adapter = input_adapters.InputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data to Input to be numpy" in str(info.value)


def test_input_numerical():
    x = np.array([[["unknown"]]])
    adapter = input_adapters.InputAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data to Input to be numerical" in str(info.value)


def test_text_adapt_unbatched_dataset():
    x = tf.data.Dataset.from_tensor_slices(np.array(["a b c", "b b c"]))
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)

    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)


def test_text_adapt_batched_dataset():
    x = tf.data.Dataset.from_tensor_slices(np.array(["a b c", "b b c"])).batch(32)
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)

    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)


def test_text_adapt_np():
    x = np.array(["a b c", "b b c"])
    adapter = input_adapters.TextAdapter()
    x = adapter.adapt(x, batch_size=32)

    assert data_utils.dataset_shape(x).as_list() == [None]
    assert isinstance(x, tf.data.Dataset)


def test_text_input_type_error():
    x = "unknown"
    adapter = input_adapters.TextAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data to TextInput to be numpy" in str(info.value)


def test_time_series_input_type_error():
    x = "unknown"
    adapter = input_adapters.TimeseriesAdapter()
    with pytest.raises(TypeError) as info:
        x = adapter.adapt(x, batch_size=32)
    assert "Expect the data in TimeseriesInput to be numpy" in str(info.value)


def test_time_series_input_transform_df_to_dataset():
    adapter = input_adapters.TimeseriesAdapter()

    x = adapter.adapt(pd.DataFrame(np.random.rand(100, 32)), batch_size=32)

    assert isinstance(x, tf.data.Dataset)
