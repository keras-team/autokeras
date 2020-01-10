from unittest import mock

import kerastuner
import pytest
import tensorflow as tf

from autokeras import tuner as tuner_module
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.tuner.Greedy._prepare_run')
def test_add_early_stopping(_, base_tuner_search, tmp_dir):
    hyper_graph = common.build_hyper_graph()
    hp = kerastuner.HyperParameters()
    preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
    preprocess_graph.build(hp)
    keras_graph.inputs[0].shape = hyper_graph.inputs[0].shape
    tuner = tuner_module.Greedy(
        hypermodel=lambda hp: None,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = (mock.Mock(),)
    tuner.oracle = oracle
    mock_graph = mock.Mock()
    mock_graph.build_graphs.return_value = (mock.Mock(), mock.Mock())

    tuner.search(mock_graph)

    callbacks = base_tuner_search.call_args_list[0][1]['callbacks']
    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.__init__')
@mock.patch('autokeras.tuner.Greedy._prepare_run')
def test_overwrite_init(_, base_tuner_init, tmp_dir):
    tuner_module.Greedy(
        hypermodel=lambda hp: None,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)

    assert not base_tuner_init.call_args_list[0][1]['overwrite']


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.tuner.Greedy._prepare_run')
def test_overwrite_search(_, base_tuner_search, tmp_dir):
    hyper_graph = common.build_hyper_graph()
    hp = kerastuner.HyperParameters()
    preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
    preprocess_graph.build(hp)
    keras_graph.inputs[0].shape = hyper_graph.inputs[0].shape
    tuner = tuner_module.Greedy(
        hypermodel=lambda hp: None,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = (mock.Mock(),)
    tuner.oracle = oracle
    mock_graph = mock.Mock()
    mock_graph.build_graphs.return_value = (mock.Mock(), mock.Mock())

    tuner.search(mock_graph)

    assert tuner._finished
