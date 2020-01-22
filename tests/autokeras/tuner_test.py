from unittest import mock

import pytest
import tensorflow as tf

from autokeras import tuner as tuner_module
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
def test_add_early_stopping(base_tuner_search, tmp_dir):
    graph = common.build_graph()
    tuner = tuner_module.Greedy(
        hypermodel=graph,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = (mock.Mock(),)
    tuner.oracle = oracle

    tuner.search()

    callbacks = base_tuner_search.call_args_list[0][1]['callbacks']
    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.__init__')
def test_overwrite_init(base_tuner_init, tmp_dir):
    tuner_module.Greedy(
        hypermodel=lambda hp: None,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)

    assert not base_tuner_init.call_args_list[0][1]['overwrite']


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
def test_overwrite_search(base_tuner_search, tmp_dir):
    graph = common.build_graph()
    tuner = tuner_module.Greedy(
        hypermodel=graph,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = (mock.Mock(),)
    tuner.oracle = oracle

    tuner.search()

    assert tuner._finished
