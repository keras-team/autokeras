# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
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
        oracle: kerastuner Oracle.
        hypermodel: kerastuner KerasHyperModel.
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        # Save or load the HyperModel.
        self.hypermodel.hypermodel.save(os.path.join(self.project_dir, "graph"))
        self.hyper_pipeline = None

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        return

    def get_best_model(self):
        with hm_module.maybe_distribute(self.distribution_strategy):
            model = tf.keras.models.load_model(self.best_model_path)
        return model

    def get_best_pipeline(self):
        return pipeline_module.load_pipeline(self.best_pipeline_path)

    def _pipeline_path(self, trial_id):
        return os.path.join(self.get_trial_dir(trial_id), "pipeline")

    def _prepare_model_build(self, hp, **kwargs):
        """Prepare for building the Keras model.

        It build the Pipeline from HyperPipeline, transform the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
        dataset = kwargs["x"]
        pipeline = self.hyper_pipeline.build(hp, dataset)
        pipeline.fit(dataset)
        dataset = pipeline.transform(dataset)
        self.hypermodel.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

        if "validation_data" in kwargs:
            validation_data = pipeline.transform(kwargs["validation_data"])
        else:
            validation_data = None
        return pipeline, dataset, validation_data

    def _on_build_begin(self, trial_id, hp, args, kwargs):
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            hp, **kwargs
        )
        pipeline.save(self._pipeline_path(trial_id))

    def _on_train_begin(self, model, hp, args, kwargs):
        """Adapt the preprocessing layers and tune the fit arguments."""
        x = kwargs["x"]
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
                temp_x = temp_x.map(layer)
                layer = get_output_layer(layer.output)
        return model

    def search(self, epochs=None, callbacks=None, validation_split=0, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Insert early-stopping for adaptive number of epochs.
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )

        # Insert early-stopping for acceleration.
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            new_callbacks.append(
                tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            )

        # Populate initial search space.
        hp = self.oracle.get_space()
        self._prepare_model_build(hp, **fit_kwargs)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

        super().search(epochs=epochs, callbacks=new_callbacks, **fit_kwargs)

        # Train the best model use validation data.
        # Train the best model with enough number of epochs.
        if validation_split > 0 or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            copied_fit_kwargs["epochs"] = epochs
            if not epochs_provided:
                copied_fit_kwargs["epochs"] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if validation_split > 0:
                copied_fit_kwargs["x"] = copied_fit_kwargs["x"].concatenate(
                    fit_kwargs["validation_data"]
                )
                copied_fit_kwargs.pop("validation_data")

            self.hypermodel.hypermodel.set_fit_args(
                0, epochs=copied_fit_kwargs["epochs"]
            )
            pipeline, model = self.final_fit(**copied_fit_kwargs)
        else:
            model = self.get_best_models()[0]
            pipeline = pipeline_module.load_pipeline(
                self._pipeline_path(self.oracle.get_best_trials(1)[0].trial_id)
            )

        model.save(self.best_model_path)
        pipeline.save(self.best_pipeline_path)
        self._finished = True

    def get_state(self):
        state = super().get_state()
        state.update({"finished": self._finished})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get("finished")

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [
            copy.deepcopy(callbacks)
            for callback in callbacks
            if not isinstance(callback, tf_callbacks.EarlyStopping)
        ]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        # steps counts from 0, so epochs = step + 1.
        return self.oracle.get_trial(best_trial.trial_id).best_step + 1

    def _build_best_model(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        return self.hypermodel.build(best_hp)

    def final_fit(self, **kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            best_hp, **kwargs
        )

        model = self._build_best_model()
        self.adapt(model, kwargs["x"])
        model.fit(**kwargs)
        return pipeline, model

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, "best_model")

    @property
    def best_pipeline_path(self):
        return os.path.join(self.project_dir, "best_pipeline")

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials
