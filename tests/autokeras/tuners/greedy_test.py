# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
