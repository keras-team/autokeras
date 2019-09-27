from unittest import mock

import pytest
import tensorflow as tf

import autokeras as ak
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


def mock_run_trial_for_early_stopping(trial, hp, fit_args, fit_kwargs):
    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in fit_kwargs['callbacks']])


@mock.patch('autokeras.tuner.AutoTuner.run_trial',
            side_effect=mock_run_trial_for_early_stopping)
@mock.patch('autokeras.tuner.AutoTuner.on_trial_end',
            side_effect=common.do_nothing)
def test_add_early_stopping(_, _1, tmp_dir):
    input_shape = (32,)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape)
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes)

    input_node = ak.Input(shape=input_shape)
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.ClassificationHead(output_shape=(num_classes,))(output_node)
    hypermodel = ak.hypermodel.graph.HyperBuiltGraphHyperModel(input_node,
                                                               output_node)
    tuner = ak.tuner.RandomSearch(
        hypermodel=hypermodel,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    tuner.search(x, y, validation_data=(x, y))
