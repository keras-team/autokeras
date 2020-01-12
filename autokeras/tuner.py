import copy
import os

import kerastuner
import kerastuner.engine.hypermodel as hm_module
import tensorflow as tf

from autokeras import oracle as oracle_module


class AutoTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    The AutoTuner uses EarlyStopping for acceleration during the search and fully
    train the model with full epochs and with both training and validation data.
    The fully trained model is the best model to be used by AutoModel.

    # Arguments
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._finished = False
        # hyper_graph is set during fit of AutoModel.
        self.hyper_graph = None
        self.preprocess_graph = None

    # Override the function to prevent building the model during initialization.
    def _populate_initial_space(self):
        pass

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

    def search(self,
               hyper_graph,
               callbacks=None,
               fit_on_val_data=False,
               **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            hyper_graph: HyperGraph. The HyperGraph to be tuned.
            callbacks: A list of callback functions. Defaults to None.
            fit_on_val_data: Boolean. Use the training set and validation set for the
                final fit of the best model.
        """
        if self._finished:
            return
        self.hyper_graph = hyper_graph

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
                    for callback in callbacks]) or fit_on_val_data:
            best_trial = self.oracle.get_best_trials(1)[0]
            best_hp = best_trial.hyperparameters
            preprocess_graph, keras_graph = self.hyper_graph.build_graphs(best_hp)
            fit_kwargs['callbacks'] = self._deepcopy_callbacks(callbacks)
            self._prepare_run(preprocess_graph, fit_kwargs, fit=True)
            if fit_on_val_data:
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


class Greedy(AutoTuner):

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 initial_hps=None,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        self.seed = seed
        oracle = oracle_module.GreedyOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super().__init__(
            hypermodel=hypermodel,
            oracle=oracle,
            **kwargs)

    def search(self, hyper_graph, **kwargs):
        self.oracle.hyper_graph = hyper_graph
        super().search(hyper_graph=hyper_graph, **kwargs)


INITIAL_HPS = {
    'image_classifier': [{
        'image_block_1/block_type':
        'vanilla', 'image_block_1/normalize': True,
        'image_block_1/augment': False,
        'image_block_1_vanilla/kernel_size': 3,
        'image_block_1_vanilla/num_blocks': 2,
        'image_block_1_vanilla/separable': True,
        'image_block_1_vanilla/filters_0_1': 32,
        'image_block_1_vanilla/filters_0_2': 32,
        'image_block_1_vanilla/filters_1_1': 32,
        'image_block_1_vanilla/filters_1_2': 16,
        'spatial_reduction_1/reduction_type': 'global_avg',
        'dense_block_1/num_layers': 2,
        'dense_block_1/use_batchnorm': False,
        'dense_block_1/dropout_rate': 0,
        'dense_block_1/units_0': 32,
        'dense_block_1/units_1': 32,
        'optimizer': 'adam'
    }],
}


class ImageClassifierTuner(Greedy):
    def __init__(self, **kwargs):
        super().__init__(
            initial_hps=INITIAL_HPS['image_classifier'],
            **kwargs)


TUNER_CLASSES = {
    'bayesian': BayesianOptimization,
    'random': RandomSearch,
    'hyperband': Hyperband,
    'greedy': Greedy,
    'image_classifier': ImageClassifierTuner,
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
