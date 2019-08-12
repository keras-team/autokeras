import os
import copy
import inspect

import kerastuner
import tensorflow as tf


class AutoTuner(kerastuner.Tuner):
    """Modified KerasTuner base class to include preprocessing layers."""

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        # Initialize new fit kwargs for the current trial.
        new_fit_kwargs = copy.copy(fit_kwargs)
        new_fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))

        # Preprocess the dataset and set the shapes of the HyperNodes.
        self.hypermodel.hyper_build(hp)
        dataset, validation_data = self.hypermodel.preprocess(
            hp,
            new_fit_kwargs.get('x', None),
            new_fit_kwargs.get('validation_data', None),
            fit=True)
        self._save_preprocessors(trial.trial_id, trial.directory)

        # Batching
        batch_size = new_fit_kwargs.get('batch_size', 32)
        dataset = dataset.batch(batch_size)
        validation_data = validation_data.batch(batch_size)

        # Update the new fit kwargs values
        new_fit_kwargs['x'] = dataset
        new_fit_kwargs['validation_data'] = validation_data
        new_fit_kwargs['batch_size'] = None
        new_fit_kwargs['y'] = None

        super().run_trial(trial, hp, [], new_fit_kwargs)

    def get_best_hp(self, num_models=1):
        """Returns hyperparameters used to build the best model(s).

        # Arguments
            num_models (int, optional): Number of best models, whose building
                HyperParameters to return. Models will be returned in sorted order
                starting from the best. Defaults to 1.

        # Returns
            List of HyperParameter instances.
        """
        best_trials = self._get_best_trials(num_models)
        return [trial.hyperparameters.copy()
                for trial in best_trials]

    def _save_preprocessors(self, trial_id, base_directory='.'):
        filename = '%s-preprocessors' % trial_id
        path = os.path.join(base_directory, filename)
        self.hypermodel.save_preprocessors(path)

    def get_best_trials(self, num_trials=1):
        return super()._get_best_trials(num_trials)

    def load_trial(self, trial):
        self.hypermodel.hyper_build(trial.hyperparameters)
        filename = '%s-preprocessors' % trial.trial_id
        path = os.path.join(trial.directory, filename)
        self.hypermodel.load_preprocessors(path)


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class HyperBand(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
