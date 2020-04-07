from unittest import mock

import tensorflow as tf

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
    print(callbacks)
    assert not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks])
