import copy
import random

import kerastuner
import numpy as np

from autokeras import blocks
from autokeras import graph
from autokeras.engine import tuner as tuner_module


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

    HYPER = 'HYPER'
    PREPROCESS = 'PREPROCESS'
    OPT = 'OPT'
    ARCH = 'ARCH'
    STAGES = [HYPER, PREPROCESS, OPT, ARCH]

    @staticmethod
    def next_stage(stage):
        stages = GreedyOracle.STAGES
        return stages[(stages.index(stage) + 1) % len(stages)]

    def __init__(self,
                 hypermodel,
                 initial_hps=None,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.initial_hps = initial_hps or []
        self._tried_initial_hps = [False] * len(self.initial_hps)
        self.hypermodel = hypermodel
        # Sets of HyperParameter names.
        self._hp_names = {
            GreedyOracle.HYPER: set(),
            GreedyOracle.PREPROCESS: set(),
            GreedyOracle.OPT: set(),
            GreedyOracle.ARCH: set(),
        }
        # The quota used to tune each category of hps.
        self.seed = seed or random.randint(1, 1e4)
        # Incremented at every call to `populate_space`.
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._max_collisions = 5

    def get_state(self):
        state = super().get_state()
        new_hp_names = {}
        for key, value in self._hp_names.items():
            new_hp_names[key] = list(value)
        state.update({
            'initial_hps': self.initial_hps,
            'tried_initial_hps': self._tried_initial_hps,
            'hypermodel': self.hypermodel.get_config(),
            'hp_names': new_hp_names,
            'seed': self.seed,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
            'max_collisions': self._max_collisions,
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state['initial_hps']
        self._tried_initial_hps = state['tried_initial_hps']
        self.hypermodel = graph.Graph.from_config(state['hypermodel'])
        new_hp_names = state['hp_names']
        for key, value in new_hp_names.items():
            self._hp_names[key] = set(value)
        self.seed = state['seed']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])
        self._max_collisions = state['max_collisions']

    def update_space(self, hyperparameters):
        # Get the block names.
        self.hypermodel.build(hyperparameters)

        # Add the new Hyperparameters to different categories.
        ref_names = {hp.name for hp in self.hyperparameters.space}
        for hp in hyperparameters.space:
            if hp.name not in ref_names:
                hp_type = None
                if any([hp.name.startswith(block.name)
                        for block in self.hypermodel.blocks if isinstance(block, (
                            blocks.ImageBlock,
                            blocks.TextBlock,
                            blocks.StructuredDataBlock,
                        ))]):
                    hp_type = GreedyOracle.HYPER
                elif any([hp.name.startswith(block.name)
                          for block in self.hypermodel.blocks if isinstance(block, (
                              blocks.TextToIntSequence,
                              blocks.TextToNgramVector,
                              blocks.Normalization,
                              blocks.ImageAugmentation,
                              blocks.CategoricalToNumerical
                          ))]):
                    hp_type = GreedyOracle.PREPROCESS
                elif any([hp.name.startswith(block.name)
                          for block in self.hypermodel.blocks if isinstance(block, (
                              blocks.Embedding,
                              blocks.ConvBlock,
                              blocks.RNNBlock,
                              blocks.DenseBlock,
                              blocks.ResNetBlock,
                              blocks.XceptionBlock,
                              blocks.Merge,
                              blocks.Flatten,
                              blocks.SpatialReduction,
                              blocks.TemporalReduction,
                              blocks.ClassificationHead,
                              blocks.RegressionHead,
                          ))]):
                    hp_type = GreedyOracle.ARCH
                else:
                    hp_type = GreedyOracle.OPT
                self._hp_names[hp_type].add(hp.name)

        super().update_space(hyperparameters)

    def _generate_stage(self):
        probabilities = np.array([pow(len(value), 2)
                                  for value in self._hp_names.values()])
        sum_p = np.sum(probabilities)
        if sum_p == 0:
            probabilities = np.array([1] * len(probabilities))
            sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        return np.random.choice(list(self._hp_names.keys()), p=probabilities)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    def _populate_space(self, trial_id):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            hp = self.get_space()
            # while not all initial_hps are registered in hp.
            while len(set(values.keys()) - set(hp.values.keys())) != 0:
                hp.values = copy.copy(values)
                self.hypermodel.build(hp)
                self.update_space(hp)
                hp = self.get_space()
            return {'status': kerastuner.engine.trial.TrialStatus.RUNNING,
                    'values': values}

        stage = self._generate_stage()
        for _ in range(len(GreedyOracle.STAGES)):
            values = self._generate_stage_values(stage)
            # Reached max collisions.
            if values is None:
                # Try next stage.
                stage = GreedyOracle.next_stage(stage)
                continue
            # Values found.
            return {'status': kerastuner.engine.trial.TrialStatus.RUNNING,
                    'values': values}
        # All stages reached max collisions.
        return {'status': kerastuner.engine.trial.TrialStatus.STOPPED,
                'values': None}

    def _generate_stage_values(self, stage):
        best_trials = self.get_best_trials()
        if best_trials:
            best_values = best_trials[0].hyperparameters.values
        else:
            best_values = self.hyperparameters.values
        collisions = 0
        while True:
            # Generate new values for the current stage.
            values = {}
            for p in self.hyperparameters.space:
                if p.name in self._hp_names[stage]:
                    values[p.name] = p.random_sample(self._seed_state)
                    self._seed_state += 1
            values = {**best_values, **values}
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash not in self._tried_so_far:
                self._tried_so_far.add(values_hash)
                break
            collisions += 1
            if collisions > self._max_collisions:
                # Reached max collisions. No value to return.
                return None
        return values


class Greedy(tuner_module.AutoTuner):

    def __init__(self,
                 hypermodel,
                 objective='val_loss',
                 max_trials=10,
                 initial_hps=None,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        self.seed = seed
        oracle = GreedyOracle(
            hypermodel=hypermodel,
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super().__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            **kwargs)
