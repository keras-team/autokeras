from unittest import mock

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
from tests import utils


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
def test_add_early_stopping(fit_fn, base_tuner_search, tmp_path):
    graph = utils.build_graph()
    tuner = tuner_module.AutoTuner(
        oracle=greedy.GreedyOracle(graph, objective='val_loss'),
        hypermodel=graph,
        directory=tmp_path)

    tuner.search(x=None, epochs=10)

    callbacks = base_tuner_search.call_args_list[0][1]['callbacks']
    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
def test_overwrite_search(fit_fn, base_tuner_search, tmp_path):
    graph = utils.build_graph()
    tuner = tuner_module.AutoTuner(
        oracle=greedy.GreedyOracle(graph, objective='val_loss'),
        hypermodel=graph,
        directory=tmp_path)

    tuner.search(epochs=10)

    assert tuner._finished


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.engine.tuner.AutoTuner.final_fit')
@mock.patch('autokeras.engine.tuner.AutoTuner._get_best_trial_epochs')
def test_no_epochs(best_epochs, fit_fn, base_tuner_search, tmp_path):
    best_epochs.return_value = 2
    graph = utils.build_graph()
    tuner = tuner_module.AutoTuner(
        oracle=greedy.GreedyOracle(graph, objective='val_loss'),
        hypermodel=graph,
        directory=tmp_path)

    tuner.search(x=mock.Mock(), epochs=None, fit_on_val_data=True,
                 validation_data=mock.Mock())

    callbacks = fit_fn.call_args_list[0][1]['callbacks']
    assert not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks])


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
