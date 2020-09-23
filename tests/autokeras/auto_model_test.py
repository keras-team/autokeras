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

from unittest import mock

import kerastuner
import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from tests import utils


def test_auto_model_objective_is_kt_objective(tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path
    )

    assert isinstance(auto_model.objective, kerastuner.Objective)


def test_auto_model_max_trial_field_as_specified(tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, max_trials=10
    )

    assert auto_model.max_trials == 10


def test_auto_model_directory_field_as_specified(tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path
    )

    assert auto_model.directory == tmp_path


def test_auto_model_project_name_field_as_specified(tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(),
        ak.RegressionHead(),
        directory=tmp_path,
        project_name="auto_model",
    )

    assert auto_model.project_name == "auto_model"


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_evaluate(tuner_fn, tmp_path):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    auto_model = ak.AutoModel(
        input_node, output_node, directory=tmp_path, max_trials=1
    )
    auto_model.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    auto_model.evaluate(tf.data.Dataset.from_tensor_slices((x_train, y_train)))
    assert tuner_fn.called


def get_single_io_auto_model(tmp_path):
    return ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, max_trials=2
    )


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_auto_model_predict(tuner_fn, tmp_path):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner_fn.called


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_final_fit_concat(tuner_fn, tmp_path):
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    assert tuner.search.call_args_list[0][1]["validation_split"]


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_final_fit_not_concat(tuner_fn, tmp_path):
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not tuner.search.call_args_list[0][1]["validation_split"]


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_overwrite(tuner_fn, tmp_path):
    tuner_class = tuner_fn.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not tuner_class.call_args_list[0][1]["overwrite"]


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_export_model(tuner_fn, tmp_path):
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    auto_model.export_model()
    assert tuner.get_best_model.called


def get_multi_io_auto_model(tmp_path):
    return ak.AutoModel(
        [ak.ImageInput(), ak.ImageInput()],
        [ak.RegressionHead(), ak.RegressionHead()],
        directory=tmp_path,
        max_trials=2,
        overwrite=False,
    )


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_multi_io_with_tf_dataset_doesnt_crash(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, epochs=2)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_single_nested_dataset_doesnt_crash(tuner_fn, tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(),
        ak.RegressionHead(),
        directory=tmp_path,
        max_trials=2,
        overwrite=False,
    )
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), y1))
    auto_model.fit(dataset, epochs=2)


def dataset_error(x, y, validation_data, message, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    with pytest.raises(ValueError) as info:
        auto_model.fit(x, y, epochs=2, validation_data=validation_data)
    assert message in str(info.value)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_data_io_consistency_input(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(dataset, None, dataset, "Expected x to have", tmp_path)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_data_io_consistency_output(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1,)))
    dataset_error(dataset, None, dataset, "Expected y to have", tmp_path)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_data_io_consistency_validation(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    val_dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(
        dataset, None, val_dataset, "Expected x in validation_data to have", tmp_path
    )


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_dataset_and_y(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    x = tf.data.Dataset.from_tensor_slices((x1, x1))
    y = tf.data.Dataset.from_tensor_slices((y1, y1))
    val_dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(x, y, val_dataset, "Expected y to be None", tmp_path)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_multi_input_predict(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices(((x1, x1),))
    auto_model.predict(dataset2)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_multi_input_predict2(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices((x1, x1))
    auto_model.predict(dataset2)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_single_input_predict_doesnt_crash(tuner_fn, tmp_path):
    auto_model = get_single_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices((x1, y1))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices((x1, y1))
    auto_model.predict(dataset2)


def test_invalid_tuner_name_error(tmp_path):
    with pytest.raises(ValueError) as info:
        ak.AutoModel(
            ak.ImageInput(), ak.RegressionHead(), directory=tmp_path, tuner="unknown"
        )

    assert "Expected the tuner argument to be one of" in str(info.value)


def test_no_validation_data_nor_split_error(tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path
    )
    with pytest.raises(ValueError) as info:
        auto_model.fit(
            x=np.random.rand(100, 32, 32, 3),
            y=np.random.rand(100, 1),
            validation_split=0,
        )

    assert "Either validation_data or a non-zero" in str(info.value)


@mock.patch("autokeras.auto_model.get_tuner_class")
def test_predict_tuple_x_and_tuple_y_predict_doesnt_crash(tuner_fn, tmp_path):
    auto_model = ak.AutoModel(
        ak.ImageInput(), ak.RegressionHead(), directory=tmp_path
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        ((np.random.rand(100, 32, 32, 3),), (np.random.rand(100, 1),))
    )
    auto_model.fit(dataset)
    auto_model.predict(dataset)
