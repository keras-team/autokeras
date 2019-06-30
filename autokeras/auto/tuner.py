import tensorflow as tf
import kerastuner


class AutoTuner(kerastuner.Tuner):

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        fit_kwargs.update(dict(zip(tf.keras.Model.fit.__code__.co_varnames,
                                   fit_args)))
        fit_args = []
        x, y, val_x, val_y = self.hypermodel.preprocess(hp, *fit_args, **fit_kwargs)
        fit_kwargs['x'] = x
        fit_kwargs['y'] = y
        fit_kwargs['validation_data'] = (val_x, val_y)
        super(AutoTuner, self).run_trial(trial, hp, fit_args, fit_kwargs)

    def get_best_hp(self, num_models=1):
        best_trials = self._get_best_trials(num_models)
        return [trial.hyperparameters.copy()
                for trial in best_trials]


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    pass
