import copy
import inspect
import os

import kerastuner
import tensorflow as tf


class AutoTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    # Arguments
        hyper_graph: HyperGraph. The HyperGraph to be tuned.
        **kwargs: The other args supported by KerasTuner.
    """

    def __init__(self, hyper_graph, **kwargs):
        super().__init__(**kwargs)
        self.hyper_graph = hyper_graph
        self.preprocess_graph = None
        self.need_fully_train = False
        self.best_hp = None

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

    def search(self, concat=False, *fit_args, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            concat: Boolean. Concatenate the training set and validation set for the
                final fit of the best model.
        """
        super().search(*fit_args, **fit_kwargs)

        best_trial = self.oracle.get_best_trials(1)[0]
        self.best_hp = best_trial.hyperparameters
        preprocess_graph, keras_graph, model = self.get_best_models()[0]
        preprocess_graph.save(self.best_preprocess_graph_path)
        keras_graph.save(self.best_keras_graph_path)

        # Fully train the best model with original callbacks.
        if self.need_fully_train or concat:
            new_fit_kwargs = copy.copy(fit_kwargs)
            new_fit_kwargs.update(
                dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
            if concat:
                new_fit_kwargs['x'] = new_fit_kwargs['x'].concatenate(
                    new_fit_kwargs['validation_data'][0])
            self._prepare_run(preprocess_graph, new_fit_kwargs)
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
