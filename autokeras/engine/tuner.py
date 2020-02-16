import collections
import copy
import os

import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


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

    # Override the function to prevent building the model during initialization.
    def _populate_initial_space(self):
        pass

    def get_best_model(self):
        model = super().get_best_models()[0]
        model.load_weights(self.best_model_path)
        return model

    @staticmethod
    def _adapt_model(model, dataset):
        # TODO: Remove this function after TF has fit-to-adapt feature.
        from tensorflow.keras.layers.experimental import preprocessing
        x = dataset.map(lambda x, y: x)

        def get_output_layer(tensor):
            tensor = nest.flatten(tensor)[0]
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                input_node = nest.flatten(layer.input)[0]
                if input_node is tensor:
                    return layer
            return None

        for index, input_node in enumerate(nest.flatten(model.input)):
            def get_data(*args):
                return args[index]

            temp_x = x.map(get_data)
            layer = get_output_layer(input_node)
            while isinstance(layer, preprocessing.PreprocessingLayer):
                layer.adapt(temp_x)
                layer = get_output_layer(layer.output)
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
            self._adapt_model(model, x)
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
            if fit_on_val_data:
                fit_kwargs['x'] = fit_kwargs['x'].concatenate(
                    fit_kwargs['validation_data'])
                fit_kwargs['callbacks'] = self._deepcopy_callbacks(callbacks)
            self.final_fit(**fit_kwargs)
        else:
            model = self.get_best_models()[0]

        model.save_weights(self.best_model_path)
        self._finished = True

    def final_fit(self, x=None, **fit_kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        model = self.hypermodel.build(best_hp)
        self._adapt_model(model, x)
        model.fit(x, **fit_kwargs)

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, 'best_model')

    @property
    def objective(self):
        return self.tuner.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials
