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

import kerastuner
import numpy as np

from autokeras.engine import tuner as tuner_module


class TrieNode(object):
    def __init__(self):
        super().__init__()
        self.num_leaves = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) == 0


class Trie(object):
    def __init__(self):
        super().__init__()
        self.root = TrieNode()

    def insert(self, hp_name):
        names = hp_name.split("/")

        new_word = False
        current_node = self.root
        nodes_on_path = [current_node]
        for name in names:
            if name not in current_node.children:
                current_node.children[name] = TrieNode()
                new_word = True
            current_node = current_node.children[name]
            nodes_on_path.append(current_node)
        current_node.hp_name = hp_name

        if new_word:
            for node in nodes_on_path:
                node.num_leaves += 1

    @property
    def nodes(self):
        return self._get_all_nodes(self.root)

    def _get_all_nodes(self, node):
        ret = [node]
        for key, value in node.children.items():
            ret += self._get_all_nodes(value)
        return ret

    def get_hp_names(self, node):
        if node.is_leaf():
            return [node.hp_name]
        ret = []
        for key, value in node.children.items():
            ret += self.get_hp_names(value)
        return ret


class GreedyOracle(kerastuner.Oracle):
    """An oracle combining random search and greedy algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. In each trial, it use a greedy strategy to
    generate new values for one of the categories of HyperParameters and use the best
    trial so far for the rest of the HyperParameters values.

    # Arguments
        initial_hps: A list of dictionaries in the form of
            {HyperParameter name (String): HyperParameter value}.
            Each dictionary is one set of HyperParameters, which are used as the
            initial trials for the search. Defaults to None.
        seed: Int. Random seed.
    """

    def __init__(self, initial_hps=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.initial_hps = initial_hps or []
        self._tried_initial_hps = [False] * len(self.initial_hps)

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "initial_hps": self.initial_hps,
                "tried_initial_hps": self._tried_initial_hps,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state["initial_hps"]
        self._tried_initial_hps = state["tried_initial_hps"]

    def _select_hps(self):
        trie = Trie()
        best_hps = self._get_best_hps()
        for hp in best_hps.space:
            # Not picking the fixed hps for generating new values.
            if best_hps.is_active(hp) and not isinstance(
                hp, kerastuner.engine.hyperparameters.Fixed
            ):
                trie.insert(hp.name)
        all_nodes = trie.nodes

        if len(all_nodes) <= 1:
            return []

        probabilities = np.array([1 / node.num_leaves for node in all_nodes])
        sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        node = np.random.choice(all_nodes, p=probabilities)

        return trie.get_hp_names(node)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    def _populate_space(self, trial_id):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }

        for i in range(self._max_collisions):
            hp_names = self._select_hps()
            values = self._generate_hp_values(hp_names)
            # Reached max collisions.
            if values is None:
                continue
            # Values found.
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }
        # All stages reached max collisions.
        return {
            "status": kerastuner.engine.trial.TrialStatus.STOPPED,
            "values": None,
        }

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()

    def _generate_hp_values(self, hp_names):
        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            hps = kerastuner.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    if best_hps.is_active(hp.name) and hp.name not in hp_names:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values


class Greedy(tuner_module.AutoTuner):
    def __init__(
        self,
        hypermodel,
        objective="val_loss",
        max_trials=10,
        initial_hps=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs
    ):
        self.seed = seed
        oracle = GreedyOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
