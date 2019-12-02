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


def build_hyper_graph():
    tf.keras.backend.clear_session()
    image_input = ak.ImageInput(shape=(32, 32, 3))
    merged_outputs = ak.ImageBlock()(image_input)
    head = ak.ClassificationHead(num_classes=10)
    head.output_shape = (10,)
    classification_outputs = head(merged_outputs)
    return ak.hypermodel.graph.HyperGraph(
        inputs=image_input,
        outputs=classification_outputs)


@mock.patch('kerastuner.engine.base_tuner.BaseTuner.search')
@mock.patch('autokeras.tuner.Greedy._prepare_run')
def test_add_early_stopping(_, base_tuner_search, tmp_dir):
    hyper_graph = build_hyper_graph()
    hp = kerastuner.HyperParameters()
    preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
    preprocess_graph.build(hp)
    keras_graph.inputs[0].shape = hyper_graph.inputs[0].shape
    tuner = tuner_module.Greedy(
        hyper_graph=hyper_graph,
        hypermodel=keras_graph,
        objective='val_loss',
        max_trials=1,
        directory=tmp_dir,
        seed=common.SEED)
    oracle = mock.Mock()
    oracle.get_best_trials.return_value = (mock.Mock(),)
    tuner.oracle = oracle
    mock_graph = mock.Mock()
    mock_graph.build_graphs.return_value = (mock.Mock(), mock.Mock())
    tuner.hyper_graph = mock_graph

    tuner.search()

    callbacks = base_tuner_search.call_args_list[0][1]['callbacks']
    assert any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                for callback in callbacks])


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
    for i in range(2000):
        oracle._populate_space(str(i))

    assert 'optimizer' in oracle._hp_names[tuner_module.GreedyOracle.OPT]
    assert 'classification_head_1/dropout_rate' in oracle._hp_names[
        tuner_module.GreedyOracle.ARCH]
    assert 'image_block_1/block_type' in oracle._hp_names[
        tuner_module.GreedyOracle.HYPER]
