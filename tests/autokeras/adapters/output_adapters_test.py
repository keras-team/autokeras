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

from autokeras.adapters import output_adapter
from autokeras.utils import data_utils
from tests import utils


def test_clf_from_config_fit_transform_to_dataset():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")
    adapter.fit_transform(np.array(["a", "b", "a"]))

    adapter = output_adapter.ClassificationHeadAdapter.from_config(
        adapter.get_config()
    )

    assert isinstance(adapter.transform(np.array(["a", "b", "a"])), tf.data.Dataset)


def test_clf_head_transform_pd_series_to_dataset():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    y = adapter.fit_transform(pd.read_csv(utils.TEST_CSV_PATH).pop("survived"))

    assert isinstance(y, tf.data.Dataset)


def test_clf_head_transform_df_to_dataset():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    y = adapter.fit_transform(
        pd.DataFrame(utils.generate_one_hot_labels(dtype="np", num_classes=10))
    )

    assert isinstance(y, tf.data.Dataset)


def test_clf_head_one_hot_shape_error():
    adapter = output_adapter.ClassificationHeadAdapter(name="a", num_classes=9)

    with pytest.raises(ValueError) as info:
        adapter.fit_transform(
            utils.generate_one_hot_labels(dtype="np", num_classes=10)
        )

    assert "Expect one hot encoded labels to have shape" in str(info.value)


def test_unsupported_types_error():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    with pytest.raises(TypeError) as info:
        adapter.check(1)

    assert "Expect the target data" in str(info.value)


def test_one_class_error():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    with pytest.raises(ValueError) as info:
        adapter.fit_before_convert(np.array(["a", "a", "a"]))
    assert "Expect the target data" in str(info.value)


def test_infer_ten_classes():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    adapter.fit_transform(
        utils.generate_one_hot_labels(dtype="dataset", num_classes=10)
    )

    assert adapter.num_classes == 10


def test_infer_single_column_two_classes():
    adapter = output_adapter.ClassificationHeadAdapter(name="a")

    adapter.fit(tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32))

    assert adapter.num_classes == 2


def test_specify_five_classes():
    adapter = output_adapter.ClassificationHeadAdapter(name="a", num_classes=5)

    adapter.fit(tf.data.Dataset.from_tensor_slices(np.random.rand(10, 5)).batch(32))

    assert adapter.num_classes == 5


def test_specify_two_classes_fit_single_column():
    adapter = output_adapter.ClassificationHeadAdapter(name="a", num_classes=2)

    adapter.fit(tf.data.Dataset.from_tensor_slices(np.random.rand(10, 1)).batch(32))

    assert adapter.num_classes == 2


def test_wrong_num_classes_error():
    adapter = output_adapter.ClassificationHeadAdapter(name="a", num_classes=5)

    with pytest.raises(ValueError) as info:
        adapter.fit(
            tf.data.Dataset.from_tensor_slices(np.random.rand(10, 3)).batch(32)
        )

    assert "Expect the target data for a to have shape" in str(info.value)


def test_multi_label_two_classes_has_two_columns():
    adapter = output_adapter.ClassificationHeadAdapter(name="a", multi_label=True)

    y = adapter.fit_transform(np.random.rand(10, 2))

    assert data_utils.dataset_shape(y).as_list() == [None, 2]


def test_multi_label_postprocess_to_one_hot_labels():
    y = np.random.rand(10, 3)
    adapter = output_adapter.ClassificationHeadAdapter(name="a", multi_label=True)
    adapter.fit_transform(y)

    y = adapter.postprocess(y)

    assert set(y.flatten().tolist()) == set([1, 0])


def test_reg_head_transform_pd_series():
    adapter = output_adapter.RegressionHeadAdapter(name="a")

    y = adapter.fit_transform(pd.read_csv(utils.TEST_CSV_PATH).pop("survived"))

    assert isinstance(y, tf.data.Dataset)


def test_reg_head_transform_1d_np():
    adapter = output_adapter.RegressionHeadAdapter(name="a")

    y = adapter.fit_transform(np.random.rand(10))

    assert isinstance(y, tf.data.Dataset)
