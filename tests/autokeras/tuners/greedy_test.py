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
import tensorflow as tf

import autokeras as ak
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
from tests import utils


def test_greedy_oracle_get_state_update_space_can_run():
    oracle = greedy.GreedyOracle(objective="val_loss")
    oracle.set_state(oracle.get_state())
    hp = kerastuner.HyperParameters()
    hp.Boolean("test")
    oracle.update_space(hp)


@mock.patch("autokeras.tuners.greedy.GreedyOracle.get_best_trials")
def test_greedy_oracle_populate_different_values(get_best_trials):
    hp = kerastuner.HyperParameters()
    utils.build_graph().build(hp)

    oracle = greedy.GreedyOracle(objective="val_loss", seed=utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]

    oracle.update_space(hp)
    values_a = oracle._populate_space("a")["values"]
    values_b = oracle._populate_space("b")["values"]

    assert not all([values_a[key] == values_b[key] for key in values_a])


@mock.patch("autokeras.tuners.greedy.GreedyOracle.get_best_trials")
def test_greedy_oracle_populate_doesnt_crash_with_init_hps(get_best_trials):
    hp = kerastuner.HyperParameters()
    tf.keras.backend.clear_session()
    input_node = ak.ImageInput(shape=(32, 32, 3))
    input_node.batch_size = 32
    input_node.num_samples = 1000
    output_node = ak.ImageBlock()(input_node)
    head = ak.ClassificationHead(num_classes=10)
    head.shape = (10,)
    output_node = head(output_node)
    graph = ak.graph.Graph(inputs=input_node, outputs=output_node)
    graph.build(hp)

    oracle = greedy.GreedyOracle(
        initial_hps=task_specific.IMAGE_CLASSIFIER,
        objective="val_loss",
        seed=utils.SEED,
    )
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]

    for i in range(10):
        tf.keras.backend.clear_session()
        values = oracle._populate_space("a")["values"]
        hp = oracle.hyperparameters.copy()
        hp.values = values
        graph.build(hp)
        oracle.update_space(hp)


@mock.patch("autokeras.tuners.greedy.GreedyOracle._compute_values_hash")
@mock.patch("autokeras.tuners.greedy.GreedyOracle.get_best_trials")
def test_greedy_oracle_stop_reach_max_collision(
    get_best_trials, compute_values_hash
):

    hp = kerastuner.HyperParameters()
    utils.build_graph().build(hp)

    oracle = greedy.GreedyOracle(objective="val_loss", seed=utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]
    compute_values_hash.return_value = 1

    oracle.update_space(hp)
    oracle._populate_space("a")["values"]
    assert (
        oracle._populate_space("b")["status"]
        == kerastuner.engine.trial.TrialStatus.STOPPED
    )


@mock.patch("autokeras.tuners.greedy.GreedyOracle.get_best_trials")
def test_greedy_oracle_populate_space_with_no_hp(get_best_trials):
    hp = kerastuner.HyperParameters()

    oracle = greedy.GreedyOracle(objective="val_loss", seed=utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]

    oracle.update_space(hp)
    values_a = oracle._populate_space("a")["values"]

    assert len(values_a) == 0
