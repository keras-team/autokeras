import copy
import inspect
import os

import kerastuner
import tensorflow as tf


class AutoTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    """Modified KerasTuner base class to include preprocessing layers."""

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
        dataset, validation_data = self.preprocess_graph.preprocess(
            dataset=new_fit_kwargs.get('x', None),
            validation_data=new_fit_kwargs.get('validation_data', None),
            fit=True)

        # # Batching
        batch_size = new_fit_kwargs.get('batch_size', 32)
        dataset = dataset.batch(batch_size)
        validation_data = validation_data.batch(batch_size)

        # Update the new fit kwargs values
        new_fit_kwargs['x'] = dataset
        new_fit_kwargs['validation_data'] = validation_data
        new_fit_kwargs['batch_size'] = None
        new_fit_kwargs['y'] = None

        super().run_trial(trial, **new_fit_kwargs)

    def _get_save_path(self, trial, name):
        filename = '{trial_id}-{name}'.format(trial_id=trial.trial_id, name=name)
        return os.path.join(self.get_trial_dir(trial.trial_id), filename)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)

        self.preprocess_graph.save(self._get_save_path(trial, 'preprocess_graph'))
        self.hypermodel.save(self._get_save_path(trial, 'keras_graph'))

        self.preprocess_graph = None
        self.hypermodel = None

    def load_model(self, trial):
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            trial.hyperparameters)
        preprocess_graph.reload(self._get_save_path(trial, 'preprocess_graph'))
        keras_graph.reload(self._get_save_path(trial, 'keras_graph'))
        self.hypermodel = keras_graph
        models = (preprocess_graph, keras_graph, super().load_model(trial))
        self.hypermodel = None
        return models

    def get_best_model(self):
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(
            self.best_hp)
        preprocess_graph.reload(self.best_preprocess_graph_path)
        keras_graph.reload(self.best_keras_graph_path)
        model = keras_graph.build(self.best_hp)
        model.load_weights(self.best_model_path)
        return preprocess_graph, model

    def search(self, *fit_args, **fit_kwargs):
        super().search(*fit_args, **fit_kwargs)

        best_trial = self.oracle.get_best_trials(1)[0]
        self.best_hp = best_trial.hyperparameters
        preprocess_graph, keras_graph, model = self.get_best_models()[0]
        preprocess_graph.save(self.best_preprocess_graph_path)
        keras_graph.save(self.best_keras_graph_path)

        # Fully train the best model with original callbacks.
        if self.need_fully_train:
            fit_kwargs.update(
                dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
            fit_kwargs['epochs'] = fit_kwargs['epochs'] - \
                self._get_trained_epochs(best_trial)
            model.fit(**fit_kwargs)

        model.save_weights(self.best_model_path)

    def _get_trained_epochs(self, trial):
        return len(next(iter(trial.metrics.metrics.values())).get_history())

    def _inject_callbacks(self, callbacks, trial, execution=0):
        callbacks = super()._inject_callbacks(callbacks, trial, execution)
        if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in callbacks]):
            self.need_fully_train = True
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10))
        return callbacks

    def _create_and_run_trial(self, trial_id, hp, callbacks, **fit_kwargs):
        trial = kerastuner.engine.trial.Trial(
            trial_id=trial_id,
            hyperparameters=hp.copy(),
            max_executions=self.executions_per_trial,
            base_directory=self._host.results_dir
        )
        self.trials.append(trial)
        self.on_trial_begin(trial)
        self.run_trial(trial=trial,
                       hp=hp,
                       callbacks=callbacks,
                       **fit_kwargs)
        self.on_trial_end(trial)

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
