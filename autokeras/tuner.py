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
        # TODO: Always do it in the fit mode. Different hp may result in different
        # input data for certain preprocessor.
        dataset, validation_data = self.hypermodel.preprocess(
            hp,
            new_fit_kwargs.get('x', None),
            new_fit_kwargs.get('validation_data', None))
        self.hypermodel.set_node_shapes(dataset)

        # Batching
        batch_size = new_fit_kwargs.get('batch_size', 32)
        dataset = dataset.batch(batch_size)
        validation_data = validation_data.batch(batch_size)

        # Update the new fit kwargs values
        new_fit_kwargs['x'] = dataset
        new_fit_kwargs['validation_data'] = validation_data
        new_fit_kwargs['batch_size'] = None
        new_fit_kwargs['y'] = None

        super(AutoTuner, self).run_trial(trial, hp, [], new_fit_kwargs)

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


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class HyperBand(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
