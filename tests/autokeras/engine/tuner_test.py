from unittest import mock

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
from tests import utils


def called_with_early_stopping(func):
    callbacks = func.call_args_list[0][1]['callbacks']
    return any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
def test_final_fit_with_specified_epochs(
        final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(x=None, epochs=10)

    assert final_fit.call_args_list[0][1]['epochs'] == 10


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
def test_tuner_call_super_with_early_stopping(
        final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(x=None, epochs=10)

    assert called_with_early_stopping(super_search)


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner.get_best_models',
            return_value=[mock.Mock()])
def test_no_final_fit_without_epochs_and_fov(
        get_best_models, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(x=None, epochs=None, fit_on_val_data=False)

    final_fit.assert_not_called()


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner._get_best_trial_epochs',
            return_value=2)
def test_final_fit_best_epochs_if_epoch_unspecified(
        best_epochs, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(x=mock.Mock(),
                 epochs=None,
                 fit_on_val_data=True,
                 validation_data=mock.Mock())

    assert final_fit.call_args_list[0][1]['epochs'] == 2


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner._get_best_trial_epochs',
            return_value=2)
def test_super_with_1k_epochs_if_epoch_unspecified(
        best_epochs, final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(x=mock.Mock(),
                 epochs=None,
                 fit_on_val_data=True,
                 validation_data=mock.Mock())

    assert super_search.call_args_list[0][1]['epochs'] == 1000
    assert called_with_early_stopping(super_search)


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
def test_tuner_not_call_super_search_with_overwrite(
        final_fit, super_search, tmp_path):
    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)

    tuner.search(epochs=10)
    tuner.save()
    super_search.reset_mock()

    tuner = greedy.Greedy(
        hypermodel=utils.build_graph(),
        directory=tmp_path)
    tuner.search(epochs=10)

    super_search.assert_not_called()


def test_preprocessing_adapt():
    class MockLayer(preprocessing.TextVectorization):
        def adapt(self, *args, **kwargs):
            super().adapt(*args, **kwargs)
            self.is_called = True

    (x_train, y_train), (x_test, y_test) = utils.imdb_raw()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    layer1 = MockLayer(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=40)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(layer1)
    model.add(tf.keras.layers.Embedding(50001, 10))
    model.add(tf.keras.layers.Dense(1))

    tuner_module.AutoTuner.adapt(model, dataset)

    assert layer1.is_called
