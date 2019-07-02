import tensorflow as tf
import kerastuner


class AutoTuner(kerastuner.Tuner):
    """Modified KerasTuner base class to include preprocessing layers."""

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        """Preprocess the x and y before calling the base run_trial."""
        fit_kwargs.update(dict(zip(tf.keras.Model.fit.__code__.co_varnames,
                                   fit_args)))
        fit_args = []
        x, y, validation_data = self.hypermodel.preprocess(hp,
                                                           *fit_args,
                                                           **fit_kwargs)
        fit_kwargs['x'] = x
        fit_kwargs['y'] = y
        fit_kwargs['validation_data'] = validation_data
        super(AutoTuner, self).run_trial(trial, hp, fit_args, fit_kwargs)

    def get_best_hp(self, num_models=1):
        """Returns hyperparameters used to build the best model(s).

        Args:
            num_models (int, optional): Number of best models, whose building
                HyperParameters to return. Models will be returned in sorted order
                starting from the best. Defaults to 1.

        Returns:
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
