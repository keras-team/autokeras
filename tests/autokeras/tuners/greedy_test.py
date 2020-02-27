from unittest import mock

import kerastuner

from autokeras.tuners import greedy
from tests import utils


def test_random_oracle_state():
    graph = utils.build_graph()
    oracle = greedy.GreedyOracle(
        hypermodel=graph,
        objective='val_loss',
    )
    oracle.hypermodel = graph
    oracle.set_state(oracle.get_state())
    assert oracle.hypermodel is graph


@mock.patch('autokeras.tuners.greedy.GreedyOracle.get_best_trials')
def test_random_oracle(fn):
    graph = utils.build_graph()
    oracle = greedy.GreedyOracle(
        hypermodel=graph,
        objective='val_loss',
    )
    oracle.hypermodel = graph
    trial = mock.Mock()
    hp = kerastuner.HyperParameters()
    trial.hyperparameters = hp
    fn.return_value = [trial]

    oracle.update_space(hp)
    for i in range(2000):
        oracle._populate_space(str(i))

    assert 'optimizer' in oracle._hp_names[greedy.GreedyOracle.OPT]
    assert 'classification_head_1/dropout_rate' in oracle._hp_names[
        greedy.GreedyOracle.ARCH]
    assert 'image_block_1/block_type' in oracle._hp_names[
        greedy.GreedyOracle.HYPER]
