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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from autokeras import keras_layers
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
from tests import utils


def called_with_early_stopping(func):
    callbacks = func.call_args_list[0][1]["callbacks"]
    return any(
        [
            isinstance(callback, tf.keras.callbacks.EarlyStopping)
            for callback in callbacks
        ]
    )


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
def test_final_fit_with_specified_epochs(_, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    final_fit.return_value = mock.Mock(), mock.Mock()

    tuner.search(x=None, epochs=10, validation_data=None)

    assert final_fit.call_args_list[0][1]["epochs"] == 10


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
def test_tuner_call_super_with_early_stopping(_, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    final_fit.return_value = mock.Mock(), mock.Mock()

    tuner.search(x=None, epochs=10, validation_data=None)

    assert called_with_early_stopping(super_search)


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch(
    "autokeras.engine.tuner.AutoTuner.get_best_models", return_value=[mock.Mock()]
)
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
@mock.patch("autokeras.pipeline.load_pipeline")
@mock.patch("kerastuner.Oracle.get_best_trials", return_value=[mock.Mock()])
def test_no_final_fit_without_epochs_and_fov(
    _, _1, _2, get_best_models, final_fit, super_search, tmp_path
):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)

    tuner.search(x=None, epochs=None, validation_data=None)

    final_fit.assert_not_called()


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch(
    "autokeras.engine.tuner.AutoTuner._get_best_trial_epochs", return_value=2
)
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
def test_final_fit_best_epochs_if_epoch_unspecified(
    _, best_epochs, final_fit, super_search, tmp_path
):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    final_fit.return_value = mock.Mock(), mock.Mock()

    tuner.search(
        x=mock.Mock(), epochs=None, validation_split=0.2, validation_data=mock.Mock()
    )

    assert final_fit.call_args_list[0][1]["epochs"] == 2


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch(
    "autokeras.engine.tuner.AutoTuner._get_best_trial_epochs", return_value=2
)
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
def test_super_with_1k_epochs_if_epoch_unspecified(
    _, best_epochs, final_fit, super_search, tmp_path
):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    final_fit.return_value = mock.Mock(), mock.Mock()

    tuner.search(
        x=mock.Mock(), epochs=None, validation_split=0.2, validation_data=mock.Mock()
    )

    assert super_search.call_args_list[0][1]["epochs"] == 1000
    assert called_with_early_stopping(super_search)


@mock.patch("kerastuner.engine.base_tuner.BaseTuner.search")
@mock.patch("autokeras.engine.tuner.AutoTuner.final_fit")
@mock.patch("autokeras.engine.tuner.AutoTuner._prepare_model_build")
def test_tuner_not_call_super_search_with_overwrite(
    _, final_fit, super_search, tmp_path
):
    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    final_fit.return_value = mock.Mock(), mock.Mock()

    tuner.search(x=None, epochs=10, validation_data=None)
    tuner.save()
    super_search.reset_mock()

    tuner = greedy.Greedy(hypermodel=utils.build_graph(), directory=tmp_path)
    tuner.search(x=None, epochs=10, validation_data=None)

    super_search.assert_not_called()


def test_tuner_does_not_crash_with_distribution_strategy(tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path,
        distribution_strategy=tf.distribute.MirroredStrategy(),
    )
    tuner.hypermodel.build(tuner.oracle.hyperparameters)


def test_preprocessing_adapt_with_cat_to_int_and_norm():
    x = np.array([["a", 5], ["b", 6]]).astype(np.unicode)
    y = np.array([[1, 2], [3, 4]]).astype(np.unicode)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(2,), dtype=tf.string))
    model.add(keras_layers.MultiCategoryEncoding(["int", "none"]))
    model.add(preprocessing.Normalization(axis=-1))

    tuner_module.AutoTuner.adapt(model, dataset)


def test_preprocessing_adapt_with_text_vec():
    class MockLayer(preprocessing.TextVectorization):
        def adapt(self, *args, **kwargs):
            super().adapt(*args, **kwargs)
            self.is_called = True

    x_train = utils.generate_text_data()
    y_train = np.random.randint(0, 2, (100,))
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    layer1 = MockLayer(max_tokens=5000, output_mode="int", output_sequence_length=40)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(layer1)
    model.add(tf.keras.layers.Embedding(50001, 10))
    model.add(tf.keras.layers.Dense(1))

    tuner_module.AutoTuner.adapt(model, dataset)

    assert layer1.is_called


def test_adapt_with_model_with_preprocessing_layer_only():
    input_node = tf.keras.Input(shape=(10,))
    output_node = tf.keras.layers.experimental.preprocessing.Normalization()(
        input_node
    )
    model = tf.keras.Model(input_node, output_node)
    greedy.Greedy.adapt(
        model,
        tf.data.Dataset.from_tensor_slices(
            (np.random.rand(100, 10), np.random.rand(100, 10))
        ).batch(32),
    )
