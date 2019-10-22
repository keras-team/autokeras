from unittest import mock

import pytest
import tensorflow as tf

from autokeras import tuner as tuner_module
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def test_add_early_stopping(tmp_dir):
    tuner = tuner_module.RandomSearch(
        hyper_graph=mock.Mock(),
        hypermodel=mock.Mock(),
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)

    callbacks = tuner._inject_callbacks([], mock.Mock())

    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


@mock.patch('autokeras.tuner.RandomSearch._prepare_run')
@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('tensorflow.keras.Model')
def test_search(_, _1, _2, tmp_dir):
    hyper_graph = mock.Mock()
    hyper_graph.build_graphs.return_value = (mock.Mock(), mock.Mock())
    tuner = tuner_module.RandomSearch(
        hyper_graph=hyper_graph,
        hypermodel=mock.Mock(),
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = [mock.Mock(), mock.Mock(), mock.Mock()]
    tuner.oracle = oracle
    tuner.preprocess_graph = mock.Mock()
    tuner.need_fully_train = True
    tuner.search(concat=True,
                 x=mock.Mock(),
                 y=mock.Mock(),
                 validation_data=[mock.Mock()],
                 epochs=5)
