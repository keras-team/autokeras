import copy
import inspect
import os

import kerastuner
import tensorflow as tf


class AutoTuner(kerastuner.Tuner):
    """Modified KerasTuner base class to include preprocessing layers."""

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        # Initialize new fit kwargs for the current trial.
        fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
        new_fit_kwargs = copy.copy(fit_kwargs)

        # Preprocess the dataset and set the shapes of the HyperNodes.
        self.hypermodel.hyper_build(trial.hyperparameters)
        dataset, validation_data = self.hypermodel.preprocess(
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

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        filename = '%s-preprocessors' % trial.trial_id
        path = os.path.join(self.get_trial_dir(trial.trial_id), filename)
        self.hypermodel.save_preprocessors(path)
        self.hypermodel.clear_preprocessors()

    def load_model(self, trial):
        self.hypermodel.hyper_build(trial.hyperparameters)
        filename = '%s-preprocessors' % trial.trial_id
        path = os.path.join(self.get_trial_dir(trial.trial_id), filename)
        self.hypermodel.load_preprocessors(path)
        super().load_model(trial)

    def search(self, *fit_args, **fit_kwargs):
        # Format the arguments.
        fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))

        # Use early-stopping during the search for acceleration.
        callbacks = fit_kwargs.pop('callbacks', [])
        callbacks_for_search = copy.copy(callbacks)
        early_stopping_in_callbacks = any([
            isinstance(callback, tf.keras.callbacks.EarlyStopping)
            for callback in callbacks])
        if not early_stopping_in_callbacks:
            callbacks_for_search.append(
                tf.keras.callbacks.EarlyStopping(patience=10))

        super().search(callbacks=callbacks_for_search, **fit_kwargs)

        # # Fully train the best model with original callbacks.
        # if not early_stopping_in_callbacks:
        #     hp = self.get_best_trials(1)[0].hyperparameters
        #     trial_id = kerastuner.engine.tuner_utils.generate_trial_id()
        #     self._create_and_run_trial(
        #         trial_id=trial_id,
        #         hp=hp,
        #         callbacks=callbacks,
        #         **fit_kwargs)

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


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class HyperBand(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
