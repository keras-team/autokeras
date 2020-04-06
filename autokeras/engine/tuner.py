import collections
import copy
import os

import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as tf_callbacks

from autokeras import graph as graph_module
from autokeras.utils import utils


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
        # Save or load the HyperModel.
        utils.save_json(os.path.join(self.project_dir, 'graph'),
                        graph_module.serialize(self.hypermodel.hypermodel))

    # Override the function to prevent building the model during initialization.
    def _populate_initial_space(self):
        pass

    def get_best_model(self):
        model = super().get_best_models()[0]
        model.load_weights(self.best_model_path)
        return model

    def run_trial(self, trial, x=None, *fit_args, **fit_kwargs):
        # TODO: Remove this function after TF has fit-to-adapt feature.
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(
                trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])
        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(fit_kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial.trial_id, execution)
            callbacks.append(
                kerastuner.engine.tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_fit_kwargs['callbacks'] = callbacks

            model = self.hypermodel.build(trial.hyperparameters)
            utils.adapt_model(model, x)
            history = model.fit(x, *fit_args, **copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)
        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step)

    def search(self,
               epochs=None,
               callbacks=None,
               fit_on_val_data=False,
               **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            fit_on_val_data: Boolean. Use the training set and validation set for the
                final fit of the best model.
        """
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        # Insert early-stopping for adaptive number of epochs.
        if epochs is None:
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(tf_callbacks.EarlyStopping(patience=10))

        # Insert early-stopping for acceleration.
        acceleration = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            acceleration = True
            new_callbacks.append(tf_callbacks.EarlyStopping(patience=10))

        super().search(epochs=epochs, callbacks=new_callbacks, **fit_kwargs)

        # Fully train the best model with original callbacks.
        if acceleration or fit_on_val_data:
            if fit_on_val_data:
                # Concatenate training and validation data.
                fit_kwargs['x'] = fit_kwargs['x'].concatenate(
                    fit_kwargs['validation_data'])
                fit_kwargs.pop('validation_data')
                # Remove early-stopping since no validation data.
                if utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                    fit_kwargs['callbacks'] = self._deepcopy_callbacks(callbacks)
                    # Use best trial number of epochs.
                    fit_kwargs['epochs'] = self._get_best_trial_epochs()
            model = self.final_fit(**fit_kwargs)
        else:
            model = self.get_best_models()[0]

        model.save_weights(self.best_model_path)
        self._finished = True

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        return len(best_trial.metrics['val_loss']._observations)

    def final_fit(self, x=None, **fit_kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        model = self.hypermodel.build(best_hp)
        utils.adapt_model(model, x)
        model.fit(x, **fit_kwargs)
        return model

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, 'best_model')

    @property
    def objective(self):
        return self.tuner.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials
