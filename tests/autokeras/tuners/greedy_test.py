from unittest import mock

import kerastuner

from autokeras import graph as graph_module
from autokeras.tuners import greedy
from tests import utils


def test_greedy_oracle_state_hypermodel_is_graph():
    oracle = greedy.GreedyOracle(
        hypermodel=utils.build_graph(),
        objective='val_loss',
    )
    oracle.set_state(oracle.get_state())
    assert isinstance(oracle.hypermodel, graph_module.Graph)


def test_greedy_oracle_get_state_update_space_can_run():
    oracle = greedy.GreedyOracle(
        hypermodel=utils.build_graph(),
        objective='val_loss',
    )
    oracle.set_state(oracle.get_state())
    hp = kerastuner.HyperParameters()
    hp.Boolean('test')
    oracle.update_space(hp)


@mock.patch('autokeras.tuners.greedy.GreedyOracle.get_best_trials')
def test_greedy_oracle(fn):
    oracle = greedy.GreedyOracle(
        hypermodel=utils.build_graph(),
        objective='val_loss',
    )
    trial = mock.Mock()
    hp = kerastuner.HyperParameters()
    trial.hyperparameters = hp
    fn.return_value = [trial]

    oracle.update_space(hp)
    for i in range(2000):
        oracle._populate_space(str(i))

    assert 'optimizer' in oracle._hp_names[greedy.GreedyOracle.OPT]
    assert 'classification_head_1/dropout' in oracle._hp_names[
        greedy.GreedyOracle.ARCH]
    assert 'image_block_1/block_type' in oracle._hp_names[
        greedy.GreedyOracle.HYPER]
