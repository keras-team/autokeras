from unittest import mock

import kerastuner
import pytest
import tensorflow as tf

import autokeras as ak
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
        fit_on_val_data=True,
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
    tuner.search(x=mock.Mock(),
                 y=mock.Mock(),
                 validation_data=[mock.Mock()],
                 epochs=5)


def build_hyper_graph():
    image_input = ak.ImageInput(shape=(32, 32, 3))
    merged_outputs = ak.ImageBlock()(image_input)
    head = ak.ClassificationHead(num_classes=10)
    head.output_shape = (10,)
    classification_outputs = head(merged_outputs)
    return ak.hypermodel.graph.HyperGraph(
        inputs=image_input,
        outputs=classification_outputs)


def test_random_oracle_state():
    hyper_graph = build_hyper_graph()
    oracle = tuner_module.GreedyOracle(
        hyper_graph=hyper_graph,
        objective='val_loss',
    )
    oracle.set_state(oracle.get_state())
    assert oracle.hyper_graph is hyper_graph


@mock.patch('autokeras.tuner.GreedyOracle.get_best_trials')
def test_random_oracle(fn):
    hyper_graph = build_hyper_graph()
    oracle = tuner_module.GreedyOracle(
        hyper_graph=hyper_graph,
        objective='val_loss',
    )
    hp = kerastuner.HyperParameters()
    preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
    preprocess_graph.build(hp)
    keras_graph.inputs[0].shape = hyper_graph.inputs[0].shape
    keras_graph.build(hp)
    trial = mock.Mock()
    trial.hyperparameters = hp
    fn.return_value = [trial]

    oracle.update_space(hp)
    for i in range(100):
        oracle._populate_space(str(i))

    assert 'optimizer' in oracle._hp_names[tuner_module.GreedyOracle.OPT]
    assert 'classification_head_2/dropout_rate' in oracle._hp_names[
        tuner_module.GreedyOracle.ARCH]
    assert 'image_block_2/block_type' in oracle._hp_names[
        tuner_module.GreedyOracle.HYPER]
