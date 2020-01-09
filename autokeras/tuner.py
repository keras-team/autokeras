import copy
import os
import random

import kerastuner
import kerastuner.engine.hypermodel as hm_module
import tensorflow as tf

from autokeras.hypermodel import base


class AutoTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    # Arguments
        **kwargs: The other args supported by KerasTuner.
    """

    def __init__(self,
                 **kwargs):
        super().__init__(
            hypermodel=lambda hp: None,
            **kwargs)
        self._finished = False
        # hyper_graph is set during fit of AutoModel.
        self.hyper_graph = None
        self.preprocess_graph = None
        self.fit_on_val_data = None

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        pass

    def compile(self,
                hyper_graph,
                fit_on_val_data=False):
        """Config the AutoTuner.

        The hyper_graph and fit_on_val_data can only be known in fit function.

        # Arguments
            hyper_graph: HyperGraph. The HyperGraph to be tuned.
            fit_on_val_data: Boolean. Use the training set and validation set for the
                final fit of the best model.
        """
        self.hyper_graph = hyper_graph
        self.fit_on_val_data = fit_on_val_data
        # Populate the initial space.
        hp = self.oracle.get_space()
        hyper_graph.build_graphs(hp)
        self.oracle.update_space(hp)

    def run_trial(self, trial, **fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        # Initialize new fit kwargs for the current trial.
        new_fit_kwargs = copy.copy(fit_kwargs)

        # Preprocess the dataset and set the shapes of the HyperNodes.
        self.preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            trial.hyperparameters)
        self.hypermodel = hm_module.KerasHyperModel(keras_graph)

        self._prepare_run(self.preprocess_graph, new_fit_kwargs, True)

        super().run_trial(trial, **new_fit_kwargs)

    def _prepare_run(self, preprocess_graph, fit_kwargs, fit=False):
        dataset, validation_data = preprocess_graph.preprocess(
            dataset=fit_kwargs.get('x', None),
            validation_data=fit_kwargs.get('validation_data', None),
            fit=fit)

        # Batching
        batch_size = fit_kwargs.pop('batch_size', 32)
        dataset = dataset.batch(batch_size)
        validation_data = validation_data.batch(batch_size)

        # Update the new fit kwargs values
        fit_kwargs['x'] = dataset
        fit_kwargs['validation_data'] = validation_data
        fit_kwargs['y'] = None

    def _get_save_path(self, trial, name):
        filename = '{trial_id}-{name}'.format(trial_id=trial.trial_id, name=name)
        return os.path.join(self.get_trial_dir(trial.trial_id), filename)

    def on_trial_end(self, trial):
        """Save and clear the hypermodel and preprocess_graph."""
        super().on_trial_end(trial)

        self.preprocess_graph.save(self._get_save_path(trial, 'preprocess_graph'))
        self.hypermodel.hypermodel.save(self._get_save_path(trial, 'keras_graph'))

        self.preprocess_graph = None
        self.hypermodel = None

    def load_model(self, trial):
        """Load the model in a history trial.

        # Arguments
            trial: Trial. The trial to be loaded.

        # Returns
            Tuple of (PreprocessGraph, KerasGraph, tf.keras.Model).
        """
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            trial.hyperparameters)
        # TODO: Use constants for these strings.
        preprocess_graph.reload(self._get_save_path(trial, 'preprocess_graph'))
        keras_graph.reload(self._get_save_path(trial, 'keras_graph'))
        self.hypermodel = hm_module.KerasHyperModel(keras_graph)
        models = (preprocess_graph, keras_graph, super().load_model(trial))
        self.hypermodel = None
        return models

    def get_best_model(self):
        """Load the best PreprocessGraph and Keras model.

        It is mainly used by the predict and evaluate function of AutoModel.

        # Returns
            Tuple of (PreprocessGraph, tf.keras.Model).
        """
        preprocess_graph, keras_graph, model = self.get_best_models()[0]
        model.load_weights(self.best_model_path)
        return preprocess_graph, model

    def search(self, callbacks=None, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.
        """
        if self._finished:
            return
        # Insert early-stopping for acceleration.
        if not callbacks:
            callbacks = []
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks]):
            new_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))

        super().search(callbacks=new_callbacks, **fit_kwargs)

        # Fully train the best model with original callbacks.
        if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks]) or self.fit_on_val_data:
            best_trial = self.oracle.get_best_trials(1)[0]
            best_hp = best_trial.hyperparameters
            preprocess_graph, keras_graph = self.hyper_graph.build_graphs(best_hp)
            fit_kwargs['callbacks'] = self._deepcopy_callbacks(callbacks)
            self._prepare_run(preprocess_graph, fit_kwargs, fit=True)
            if self.fit_on_val_data:
                fit_kwargs['x'] = fit_kwargs['x'].concatenate(
                    fit_kwargs['validation_data'])
            model = keras_graph.build(best_hp)
            model.fit(**fit_kwargs)
        else:
            preprocess_graph, keras_graph, model = self.get_best_models()[0]

        model.save_weights(self.best_model_path)
        self._finished = True

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, 'best_model')


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class Hyperband(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass


class BayesianOptimization(AutoTuner, kerastuner.BayesianOptimization):
    """KerasTuner BayesianOptimization with preprocessing layer tuning."""
    pass


class GreedyOracle(kerastuner.Oracle):
    """An oracle combining random search and greedy algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. In each trial, it use a greedy strategy to
    generate new values for one of the categories of HyperParameters and use the best
    trial so far for the rest of the HyperParameters values.

    # Arguments
        hyper_graph: HyperGraph. The hyper_graph model to be tuned.
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

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.hyper_graph = None
        # Start from tuning the hyper block hps.
        self._stage = GreedyOracle.HYPER
        # Sets of HyperParameter names.
        self._hp_names = {
            GreedyOracle.HYPER: set(),
            GreedyOracle.PREPROCESS: set(),
            GreedyOracle.OPT: set(),
            GreedyOracle.ARCH: set(),
        }
        # The quota used to tune each category of hps.
        self._capacity = {
            GreedyOracle.HYPER: 1,
            GreedyOracle.PREPROCESS: 1,
            GreedyOracle.OPT: 1,
            GreedyOracle.ARCH: 4,
        }
        self._stage_trial_count = 0
        self.seed = seed or random.randint(1, 1e4)
        # Incremented at every call to `populate_space`.
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._max_collisions = 5

    def compile(self, hyper_graph):
        self.hyper_graph = hyper_graph

    def set_state(self, state):
        super().set_state(state)
        self._stage = state['stage']
        self._capacity = state['capacity']

    def get_state(self):
        state = super().get_state()
        state.update({
            'stage': self._stage,
            'capacity': self._capacity,
        })
        return state

    def update_space(self, hyperparameters):
        # Get the block names.
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            hyperparameters)

        # Add the new Hyperparameters to different categories.
        ref_names = {hp.name for hp in self.hyperparameters.space}
        for hp in hyperparameters.space:
            if hp.name not in ref_names:
                hp_type = None
                if any([hp.name.startswith(block.name)
                        for block in self.hyper_graph.blocks
                        if isinstance(block, base.HyperBlock)]):
                    hp_type = GreedyOracle.HYPER
                elif any([hp.name.startswith(block.name)
                          for block in preprocess_graph.blocks]):
                    hp_type = GreedyOracle.PREPROCESS
                elif any([hp.name.startswith(block.name)
                          for block in keras_graph.blocks]):
                    hp_type = GreedyOracle.ARCH
                else:
                    hp_type = GreedyOracle.OPT
                self._hp_names[hp_type].add(hp.name)

        super().update_space(hyperparameters)

    def _populate_space(self, trial_id):
        for _ in range(len(GreedyOracle.STAGES)):
            values = self._generate_stage_values()
            # Reached max collisions.
            if values is None:
                # Try next stage.
                self._stage = GreedyOracle.next_stage(self._stage)
                self._stage_trial_count = 0
                continue
            # Values found.
            self._stage_trial_count += 1
            if self._stage_trial_count == self._capacity[self._stage]:
                self._stage = GreedyOracle.next_stage(self._stage)
                self._stage_trial_count = 0
            return {'status': kerastuner.engine.trial.TrialStatus.RUNNING,
                    'values': values}
        # All stages reached max collisions.
        return {'status': kerastuner.engine.trial.TrialStatus.STOPPED,
                'values': None}

    def _generate_stage_values(self):
        best_trials = self.get_best_trials()
        if best_trials:
            best_values = best_trials[0].hyperparameters.values
        else:
            best_values = self.hyperparameters.values
        collisions = 0
        while 1:
            # Generate new values for the current stage.
            values = {}
            for p in self.hyperparameters.space:
                if p.name in self._hp_names[self._stage]:
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


class Greedy(AutoTuner):

    def __init__(self,
                 objective,
                 max_trials,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        self.seed = seed
        oracle = GreedyOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super().__init__(
            oracle=oracle,
            **kwargs)

    def compile(self, hyper_graph, **kwargs):
        super().compile(hyper_graph=hyper_graph, **kwargs)
        self.oracle.compile(hyper_graph)


TUNER_CLASSES = {
    'bayesian': BayesianOptimization,
    'random': RandomSearch,
    'hyperband': Hyperband,
    'greedy': Greedy,
    'image_classifier': Greedy,
    'image_regressor': Greedy,
    'text_classifier': Greedy,
    'text_regressor': Greedy,
    'structured_data_classifier': Greedy,
    'structured_data_regressor': Greedy,
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError('The value {tuner} passed for argument tuner is invalid, '
                         'expected one of "greedy", "random", "hyperband", '
                         '"bayesian".'.format(tuner=tuner))
