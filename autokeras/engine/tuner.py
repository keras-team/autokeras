import copy
import os

import kerastuner
import tensorflow as tf
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras.utils import utils


class AutoTuner(kerastuner.engine.tuner.Tuner):
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
        preprocessors: An instance or list of `Preprocessor` objects corresponding to
            each AutoModel input, to preprocess a `tf.data.Dataset` before passing it
            to the model. Defaults to None (no external preprocessing).
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 preprocessors=None,
                 **kwargs):
        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        self.preprocessors = nest.flatten(preprocessors)
        # Save or load the HyperModel.
        self.hypermodel.hypermodel.save(os.path.join(self.project_dir, 'graph'))

    # Override the function to prevent building the model during initialization.
    def _populate_initial_space(self):
        pass

    def get_best_model(self):
        model = super().get_best_models()[0]
        model.load_weights(self.best_model_path)
        return model

    def _on_train_begin(self, model, hp, x, *args, **kwargs):
        """Adapt the preprocessing layers and tune the fit arguments."""
        self.adapt(model, x)

    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
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
            temp_x = x.map(lambda *args: nest.flatten(args)[index])
            layer = get_output_layer(input_node)
            while isinstance(layer, preprocessing.PreprocessingLayer):
                layer.adapt(temp_x)
                layer = get_output_layer(layer.output)
        return model

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
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(tf_callbacks.EarlyStopping(patience=10))

        # Insert early-stopping for acceleration.
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            new_callbacks.append(tf_callbacks.EarlyStopping(patience=10))

        # Populate initial search space.
        hp = self.oracle.get_space()
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

        super().search(epochs=epochs, callbacks=new_callbacks, **fit_kwargs)

        # Train the best model use validation data.
        # Train the best model with enought number of epochs.
        if fit_on_val_data or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs['callbacks'] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            copied_fit_kwargs['epochs'] = epochs
            if not epochs_provided:
                copied_fit_kwargs['epochs'] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if fit_on_val_data:
                copied_fit_kwargs['x'] = copied_fit_kwargs['x'].concatenate(
                    fit_kwargs['validation_data'])
                copied_fit_kwargs.pop('validation_data')

            model = self.final_fit(**copied_fit_kwargs)
        else:
            model = self.get_best_models()[0]

        model.save_weights(self.best_model_path)
        self._finished = True

    def get_state(self):
        state = super().get_state()
        state.update({
            'finished': self._finished,
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get('finished')

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [copy.deepcopy(callbacks)
                for callback in callbacks
                if not isinstance(callback, tf_callbacks.EarlyStopping)]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        return len(best_trial.metrics.metrics['val_loss']._observations)

    def final_fit(self, x=None, **fit_kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        model = self.hypermodel.build(best_hp)
        self.adapt(model, x)
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
