from unittest import mock

import pytest
import tensorflow as tf

import autokeras as ak
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def test_add_early_stopping(tmp_dir):
    tuner = ak.tuner.RandomSearch(
        hyper_graph=mock.Mock(),
        hypermodel=mock.Mock(),
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)

    callbacks = tuner._inject_callbacks([], mock.Mock())

    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])
