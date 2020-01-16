from unittest import mock

import kerastuner

from autokeras import oracle as oracle_module
from tests import common


def test_random_oracle_state():
    hyper_graph = common.build_hyper_graph()
    oracle = oracle_module.GreedyOracle(
        objective='val_loss',
    )
    oracle.hyper_graph = hyper_graph
    oracle.set_state(oracle.get_state())
    assert oracle.hyper_graph is hyper_graph


@mock.patch('autokeras.oracle.GreedyOracle.get_best_trials')
def test_random_oracle(fn):
    hyper_graph = common.build_hyper_graph()
    oracle = oracle_module.GreedyOracle(
        objective='val_loss',
    )
    hp = kerastuner.HyperParameters()
    preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
    preprocess_graph.build(hp)
    keras_graph.inputs[0].shape = hyper_graph.inputs[0].shape
    keras_graph.build(hp)
    oracle.hyper_graph = hyper_graph
    trial = mock.Mock()
    trial.hyperparameters = hp
    fn.return_value = [trial]

    oracle.update_space(hp)
    for i in range(2000):
        oracle._populate_space(str(i))

    assert 'optimizer' in oracle._hp_names[oracle_module.GreedyOracle.OPT]
    assert 'classification_head_1/dropout_rate' in oracle._hp_names[
        oracle_module.GreedyOracle.ARCH]
    assert 'image_block_1/block_type' in oracle._hp_names[
        oracle_module.GreedyOracle.HYPER]
