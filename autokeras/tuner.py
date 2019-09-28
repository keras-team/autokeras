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

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        filename = '%s-preprocessors' % trial.trial_id
        path = os.path.join(trial.directory, filename)
        self.hypermodel.save_preprocessors(path)
        self.hypermodel.clear_preprocessors()

    def get_best_trials(self, num_trials=1):
        return super()._get_best_trials(num_trials)

    def load_trial(self, trial):
        self.hypermodel.hyper_build(trial.hyperparameters)
        filename = '%s-preprocessors' % trial.trial_id
        path = os.path.join(trial.directory, filename)
        self.hypermodel.load_preprocessors(path)

    def search(self, *fit_args, **fit_kwargs):
        # Format the arguments.
        fit_kwargs.update(
            dict(zip(inspect.getfullargspec(tf.keras.Model.fit).args, fit_args)))
        fit_args = []
        fit_kwargs.setdefault('callbacks', [])
        if fit_kwargs['callbacks'] is None:
            fit_kwargs['callbacks'] = []
        fit_kwargs.setdefault('epochs', None)

        # Insert early-stopping permanently when epochs is None.
        callbacks = fit_kwargs['callbacks']
        if fit_kwargs['epochs'] is None:
            fit_kwargs['epochs'] = 1000
            if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                        for callback in callbacks]):
                callbacks = callbacks + [
                    tf.keras.callbacks.EarlyStopping(patience=10)]
        fit_kwargs['callbacks'] = callbacks

        # Insert early-stopping temporarily for the search.
        final_fit = False
        new_fit_kwargs = copy.copy(fit_kwargs)
        callbacks = new_fit_kwargs['callbacks']
        if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                    for callback in new_fit_kwargs['callbacks']]):
            callbacks = callbacks + [tf.keras.callbacks.EarlyStopping(patience=10)]
            final_fit = True
        new_fit_kwargs['callbacks'] = callbacks

        # Start the search.
        self.on_search_begin()
        for _ in range(self.max_trials):
            # Obtain unique trial ID to communicate with the oracle.
            trial_id = kerastuner.engine.tuner_utils.generate_trial_id()
            hp = self._call_oracle(trial_id)
            if hp is None:
                # Oracle triggered exit
                return
            trial = kerastuner.engine.trial.Trial(
                trial_id=trial_id,
                hyperparameters=hp.copy(),
                max_executions=self.executions_per_trial,
                base_directory=self._host.results_dir
            )
            self.trials.append(trial)
            self.on_trial_begin(trial)
            self.run_trial(trial, hp, fit_args, new_fit_kwargs)
            self.on_trial_end(trial)

        # Final fit.
        if final_fit:
            hp = self.get_best_trials(1)[0].hyperparameters
            trial = kerastuner.engine.trial.Trial(
                trial_id=trial_id,
                hyperparameters=hp.copy(),
                max_executions=self.executions_per_trial,
                base_directory=self._host.results_dir
            )
            self.trials.append(trial)
            self.on_trial_begin(trial)
            self.run_trial(trial, hp, fit_args, fit_kwargs)
            self.on_trial_end(trial)

        self.on_search_end()


class RandomSearch(AutoTuner, kerastuner.RandomSearch):
    """KerasTuner RandomSearch with preprocessing layer tuning."""
    pass


class HyperBand(AutoTuner, kerastuner.Hyperband):
    """KerasTuner Hyperband with preprocessing layer tuning."""
    pass
