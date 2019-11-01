import copy
import inspect
import os

import kerastuner
import tensorflow as tf

from autokeras.hypermodel import graph


class AutoTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    # Arguments
        hyper_graph: HyperGraph. The HyperGraph to be tuned.
        fit_on_val_data: Boolean. Use the training set and validation set for the
            final fit of the best model.
        **kwargs: The other args supported by KerasTuner.
    """

    def __init__(self, hyper_graph, fit_on_val_data=False, **kwargs):
        super().__init__(**kwargs)
        self.hyper_graph = hyper_graph
        self.preprocess_graph = None
        self.need_fully_train = False
        self.best_hp = None
        self.fit_on_val_data = fit_on_val_data

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        # Initialize new fit kwargs for the current trial.
        fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
        new_fit_kwargs = copy.copy(fit_kwargs)

        # Preprocess the dataset and set the shapes of the HyperNodes.
        self.preprocess_graph, self.hypermodel = self.hyper_graph.build_graphs(
            trial.hyperparameters)

        self._prepare_run(self.preprocess_graph, new_fit_kwargs, True)

        super().run_trial(trial, **new_fit_kwargs)

    def _prepare_run(self, preprocess_graph, fit_kwargs, fit=False):
        dataset, validation_data = preprocess_graph.preprocess(
            dataset=fit_kwargs.get('x', None),
            validation_data=fit_kwargs.get('validation_data', None),
            fit=fit)

        # Batching
        batch_size = fit_kwargs.get('batch_size', 32)
        dataset = dataset.batch(batch_size)
        validation_data = validation_data.batch(batch_size)

        # Update the new fit kwargs values
        fit_kwargs['x'] = dataset
        fit_kwargs['validation_data'] = validation_data
        fit_kwargs['batch_size'] = None
        fit_kwargs['y'] = None

    def _get_save_path(self, trial, name):
        filename = '{trial_id}-{name}'.format(trial_id=trial.trial_id, name=name)
        return os.path.join(self.get_trial_dir(trial.trial_id), filename)

    def on_trial_end(self, trial):
        """Save and clear the hypermodel and preprocess_graph."""
        super().on_trial_end(trial)

        self.preprocess_graph.save(self._get_save_path(trial, 'preprocess_graph'))
        self.hypermodel.save(self._get_save_path(trial, 'keras_graph'))

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
        preprocess_graph.reload(self._get_save_path(trial, 'preprocess_graph'))
        keras_graph.reload(self._get_save_path(trial, 'keras_graph'))
        self.hypermodel = keras_graph
        models = (preprocess_graph, keras_graph, super().load_model(trial))
        self.hypermodel = None
        return models

    def get_best_model(self):
        """Load the best PreprocessGraph and Keras model.

        It is mainly used by the predict and evaluate function of AutoModel.

        # Returns
            Tuple of (PreprocessGraph, tf.keras.Model).
        """
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            self.best_hp)
        preprocess_graph.reload(self.best_preprocess_graph_path)
        keras_graph.reload(self.best_keras_graph_path)
        model = keras_graph.build(self.best_hp)
        model.load_weights(self.best_model_path)
        return preprocess_graph, model

    def search(self, *fit_args, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.
        """
        super().search(*fit_args, **fit_kwargs)

        best_trial = self.oracle.get_best_trials(1)[0]
        self.best_hp = best_trial.hyperparameters
        preprocess_graph, keras_graph, model = self.get_best_models()[0]
        preprocess_graph.save(self.best_preprocess_graph_path)
        keras_graph.save(self.best_keras_graph_path)

        # Fully train the best model with original callbacks.
        if self.need_fully_train or self.fit_on_val_data:
            new_fit_kwargs = copy.copy(fit_kwargs)
            new_fit_kwargs.update(
                dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
            self._prepare_run(preprocess_graph, new_fit_kwargs)
            if self.fit_on_val_data:
                new_fit_kwargs['x'] = new_fit_kwargs['x'].concatenate(
                    new_fit_kwargs['validation_data'])
            model = keras_graph.build(self.best_hp)
            model.fit(**new_fit_kwargs)

        model.save_weights(self.best_model_path)

    def _inject_callbacks(self, callbacks, trial, execution=0):
        """Inject the early-stopping callback."""
        callbacks = super()._inject_callbacks(callbacks, trial, execution)
        if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks]):
            self.need_fully_train = True
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))
        return callbacks

    @property
    def best_preprocess_graph_path(self):
        return os.path.join(self.project_dir, 'best_preprocess_graph')

    @property
    def best_keras_graph_path(self):
        return os.path.join(self.project_dir, 'best_keras_graph')

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, 'best_model')


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class HyperBand(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass


class GreedyRandomOracle(kerastuner.Oracle):
    """An oracle combining random search and greedy algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. Uses union the best HyperParameters for
    each group as the overall best HyperParameters.

    # Arguments
        hyper_graph: HyperGraph. The hyper_graph model to be tuned.
    """

    HYPER = 'HYPER'
    PREPROCESS = 'PREPROCESS'
    OPT = 'OPT'
    ARCH = 'ARCH'
    STAGES = [HYPER, PREPROCESS, OPT, ARCH]
    NEXT_STAGE = {STAGES[i]: STAGES[i + 1] for i in len(STAGES) - 1}

    def __init__(self, hyper_graph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper_graph = hyper_graph
        # Start from tuning the hyper block hps.
        self._stage = GreedyRandomOracle.HYPER
        # Sets of HyperParameter names.
        self._hp_names = {
            GreedyRandomOracle.HYPER: set(),
            GreedyRandomOracle.PREPROCESS: set(),
            GreedyRandomOracle.OPT: set(),
            GreedyRandomOracle.ARCH: set(),
        }
        self._trial_id_to_stage = {}
        # Use 10% of max_trials to tune each category except architecture_hps.
        # Use the rest quota to tune the architecture_hps.
        trial_each = max(0.1 * self.max_trials, 1)
        self._end_stage[GreedyRandomOracle.HYPER] = trial_each
        self._end_stage[GreedyRandomOracle.PREPROCESS] = trial_each * 2
        self._end_stage[GreedyRandomOracle.OPT] = trial_each * 3
        self._end_stage[GreedyRandomOracle.ARCH] = self.max_trials

    def set_state(self, state):
        super().set_state(state)
        self.hyper_graph.set_state(state['hyper_graph'])

    def get_state(self):
        state = super().get_state()
        state.update({'hyper_graph': self.hyper_graph.get_state()})
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
                        if isinstance(block, graph.HyperGraph)]):
                    hp_type = GreedyRandomOracle.HYPER
                elif any([hp.name.startswith(block.name)
                          for block in preprocess_graph.blocks]):
                    hp_type = GreedyRandomOracle.PREPROCESS
                elif any([hp.name.startswith(block.name)
                          for block in keras_graph.blocks]):
                    hp_type = GreedyRandomOracle.ARCH
                else:
                    hp_type = GreedyRandomOracle.OPT
                self._hp_names[hp_type].add(hp.name)

        super().update_space(hyperparameters)

    def _populate_space(self, trial_id):
        # TODO: handle reach max collision.
        new_values = self._random_sample(self._hp_names[self._stage])
        self._trial_id_to_stage[trial_id] = self._stage
        # Update the stage.
        if len(self.trials) == self._end_stage[self._stage]:
            self._stage = GreedyRandomOracle.NEXT_STAGE[self._stage]

        # Find the best hps in history.
        values = self._get_best_values()
        values.update(new_values)
        return {'status': kerastuner.engine.trial.TrialStatus.RUNNING,
                'values': values}

    def _get_best_values(self):
        values = {p.name: p.default for p in self.hyperparameters.space}
        best_score = {}
        for trial in self.trials.values():
            if trial.status != "COMPLETED":
                continue
            stage = self._trial_id_to_stage[trial.trial_id]
            trial_values = trial.hyperparameters.values
            score = trial.score
            if self.objective.direction == 'max':
                score = -score
            if stage not in best_score or best_score[stage] > score:
                best_score[stage] = score
                for key in self._hp_names[stage]:
                    if key in trial_values:
                        values[key] = trial_values
        return values

    def _random_sample(self, names):
        collisions = 0
        while 1:
            # Generate a set of random values.
            values = {}
            for p in self.hyperparameters.space:
                if p.name in names:
                    values[p.name] = p.random_sample(self._seed_state)
                    self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions > self._max_collisions:
                    break
                continue
            self._tried_so_far.add(values_hash)
            break
        return values


class GreedyRandom(AutoTuner):

    def __init__(self,
                 hyper_graph,
                 fit_on_val_data,
                 objective,
                 max_trials,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        self.seed = seed
        oracle = GreedyRandomOracle(
            hyper_graph=hyper_graph,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        hp = oracle.get_space()
        preprocess_graph, keras_graph = hyper_graph.build_graphs(hp)
        oracle.update_space(hp)
        super().__init__(
            hyper_graph=hyper_graph,
            fit_on_val_data=fit_on_val_data,
            oracle=oracle,
            hypermodel=keras_graph,
            **kwargs)


TUNER_CLASSES = {
    'random_search': RandomSearch,
    'image_classifier': GreedyRandom,
    'image_regressor': GreedyRandom,
    'text_classifier': GreedyRandom,
    'text_regressor': GreedyRandom,
    'structured_data_classifier': GreedyRandom,
    'structured_data_regressor': GreedyRandom,
}


def get_tuner_class(name):
    return TUNER_CLASSES.get(name)
