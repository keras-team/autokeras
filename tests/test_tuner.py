from unittest import mock

import kerastuner
import pytest
import tensorflow as tf

import autokeras as ak
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('autokeras.tuner.AutoTuner.run_trial')
@mock.patch('autokeras.tuner.kerastuner.Tuner.on_search_begin')
@mock.patch('autokeras.tuner.AutoTuner.on_trial_end')
@mock.patch('autokeras.tuner.kerastuner.Tuner._get_best_trials')
@mock.patch('kerastuner.engine.trial.Trial')
def test_add_early_stopping(_2, get_trials, _1, _, run_trial, tmp_dir):
    trial = kerastuner.engine.trial.Trial()
    trial.hyperparameters = kerastuner.HyperParameters()
    get_trials.return_value = [trial]
    input_shape = (32,)
    num_instances = 100
    num_classes = 10
    x = common.generate_data(num_instances=num_instances,
                             shape=input_shape,
                             dtype='dataset')
    y = common.generate_one_hot_labels(num_instances=num_instances,
                                       num_classes=num_classes,
                                       dtype='dataset')

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
    tuner.search(x=tf.data.Dataset.zip((x, y)),
                 validation_data=(x, y),
                 epochs=20,
                 callbacks=[])

    _, kwargs = run_trial.call_args_list[0]
    callbacks = kwargs['callbacks']
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], tf.keras.callbacks.EarlyStopping)

    _, kwargs = run_trial.call_args_list[1]
    callbacks = kwargs['callbacks']
    assert len(callbacks) == 0
