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

from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import keras
import numpy as np
import tree

from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils
from autokeras.utils import utils

TUNER_CLASSES = {
    "bayesian": tuners.BayesianOptimization,
    "random": tuners.RandomSearch,
    "hyperband": tuners.Hyperband,
    "greedy": tuners.Greedy,
}


def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError(
            'Expected the tuner argument to be one of "greedy", '
            '"random", "hyperband", or "bayesian", '
            "but got {tuner}".format(tuner=tuner)
        )


class AutoModel(object):
    """A Model defined by inputs and outputs.
    AutoModel combines a HyperModel and a Tuner to tune the HyperModel.
    The user can use it in a similar way to a Keras model since it
    also has `fit()` and  `predict()` methods.

    The AutoModel has two use cases. In the first case, the user only specifies
    the input nodes and output heads of the AutoModel. The AutoModel infers the
    rest part of the model. In the second case, user can specify the high-level
    architecture of the AutoModel by connecting the Blocks with the functional
    API, which is the same as the Keras
    [functional
    API](https://keras.io/api/models/model/#with-the-functional-api).

    # Example
    ```python
        # The user only specifies the input nodes and output heads.
        import autokeras as ak
        ak.AutoModel(
            inputs=[ak.ImageInput(), ak.TextInput()],
            outputs=[ak.ClassificationHead(), ak.RegressionHead()]
        )
    ```
    ```python
        # The user specifies the high-level architecture.
        import autokeras as ak
        image_input = ak.ImageInput()
        image_output = ak.ImageBlock()(image_input)
        text_input = ak.TextInput()
        text_output = ak.TextBlock()(text_input)
        output = ak.Merge()([image_output, text_output])
        classification_output = ak.ClassificationHead()(output)
        regression_output = ak.RegressionHead()(output)
        ak.AutoModel(
            inputs=[image_input, text_input],
            outputs=[classification_output, regression_output]
        )
    ```

    # Arguments
        inputs: A list of Node instances.
            The input node(s) of the AutoModel.
        outputs: A list of Node or Head instances.
            The output node(s) or head(s) of the AutoModel.
        project_name: String. The name of the AutoModel. Defaults to
            'auto_model'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to
            100.
        directory: String. The path to a directory for storing the search
            outputs. Defaults to None, which would create a folder with the
            name of the AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        tuner: String or subclass of AutoTuner. If string, it should be one of
            'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a
            subclass of AutoTuner. Defaults to 'greedy'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
        max_model_size: Int. Maximum number of scalars in the parameters of a
            model. Models larger than this are rejected.
        **kwargs: Any arguments supported by keras_tuner.Tuner.
    """

    def __init__(
        self,
        inputs: Union[Input, List[Input]],
        outputs: Union[head_module.Head, node_module.Node, list],
        project_name: str = "auto_model",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        tuner: Union[str, Type[tuner.AutoTuner]] = "greedy",
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        **kwargs
    ):
        self.inputs = tree.flatten(inputs)
        self.outputs = tree.flatten(outputs)
        self.seed = seed
        if seed:
            np.random.seed(seed)
        # TODO: Support passing a tuner instance.
        # Initialize the hyper_graph.
        graph = self._build_graph()
        if isinstance(tuner, str):
            tuner = get_tuner_class(tuner)
        self.tuner = tuner(
            hypermodel=graph,
            overwrite=overwrite,
            objective=objective,
            max_trials=max_trials,
            directory=directory,
            seed=self.seed,
            project_name=project_name,
            max_model_size=max_model_size,
            **kwargs
        )
        self.overwrite = overwrite
        self._heads = [output_node.in_blocks[0] for output_node in self.outputs]

    @property
    def objective(self):
        return self.tuner.objective

    @property
    def max_trials(self):
        return self.tuner.max_trials

    @property
    def directory(self):
        return self.tuner.directory

    @property
    def project_name(self):
        return self.tuner.project_name

    def _assemble(self):
        """Assemble the Blocks based on the input output nodes."""
        inputs = tree.flatten(self.inputs)
        outputs = tree.flatten(self.outputs)

        middle_nodes = [
            input_node.get_block()(input_node) for input_node in inputs
        ]

        # Merge the middle nodes.
        if len(middle_nodes) > 1:
            output_node = blocks.Merge()(middle_nodes)
        else:
            output_node = middle_nodes[0]

        outputs = tree.flatten(
            [output_blocks(output_node) for output_blocks in outputs]
        )
        return graph_module.Graph(inputs=inputs, outputs=outputs)

    def _build_graph(self):
        # Using functional API.
        if all(
            [isinstance(output, node_module.Node) for output in self.outputs]
        ):
            graph = graph_module.Graph(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all(
            [isinstance(output, head_module.Head) for output in self.outputs]
        ):
            # Clear session to reset get_uid(). The names of the blocks will
            # start to count from 1 for new blocks in a new AutoModel
            # afterwards.  When initializing multiple AutoModel with Task API,
            # if not counting from 1 for each of the AutoModel, the predefined
            # hp values in task specifiec tuners would not match the names.
            keras.backend.clear_session()
            graph = self._assemble()
            self.outputs = graph.outputs
            keras.backend.clear_session()

        return graph

    def fit(
        self,
        x=None,
        y=None,
        batch_size=32,
        epochs=None,
        callbacks=None,
        validation_split=0.2,
        validation_data=None,
        verbose=1,
        **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray. Training data x.
            y: numpy.ndarray. Training data y.
            batch_size: Int. Number of samples per gradient update. Defaults to
                32.
            epochs: Int. The number of epochs to train each model during the
                search. If unspecified, by default we train for a maximum of
                1000 epochs, but we stop training if the validation loss stops
                improving for 10 epochs (unless you specified an EarlyStopping
                callback as part of the callbacks argument, in which case the
                EarlyStopping callback you specified will determine early
                stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model
                metrics on this data at the end of each epoch.  The validation
                data is selected from the last samples in the `x` and `y` data
                provided, before shuffling. This argument is not supported when
                `x` is a dataset. The best model found would be fit on the
                entire dataset including the validation data.
            validation_data: Data on which to evaluate the loss and any model
                metrics at the end of each epoch. The model will not be trained
                on this data. `validation_data` will override
                `validation_split`. The type of the validation data should be
                the same as the training data. The best model found would be
                fit on the training dataset without the validation data.
            verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar,
                2 = one line per epoch. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment). Controls the verbosity of both KerasTuner search
                and
                [keras.Model.fit](https://keras.io/api/models/model_training_apis/#fit-method)
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://keras.io/api/models/model_training_apis/#fit-method).

        # Returns
            history: A Keras History object corresponding to the best model.
                Its History.history attribute is a record of training
                loss values and metrics values at successive epochs, as well as
                validation loss values and validation metrics values (if
                applicable).
        """
        # Check validation information.
        if not validation_data and not validation_split:
            raise ValueError(
                "Either validation_data or a non-zero validation_split "
                "should be provided."
            )

        if validation_data:
            validation_split = 0

        dataset, validation_data = self._check_and_adapt(
            x=x, y=y, validation_data=validation_data
        )
        self._analyze_data(dataset)
        self._build_hyper_pipeline()

        # Split the data with validation_split.
        if validation_data is None and validation_split:
            dataset, validation_data = data_utils.split_dataset(
                dataset, validation_split
            )

        x, y = dataset
        history = self.tuner.search(
            x=x,
            y=y,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split,
            verbose=verbose,
            batch_size=batch_size,
            **kwargs
        )

        return history

    def _adapt(self, dataset, hms):
        sources = tree.flatten(dataset)
        adapted = []
        for source, hm in zip(sources, hms):
            source = hm.get_adapter().adapt(source)
            adapted.append(source)
        if len(adapted) == 1:
            return adapted[0]
        return tuple(adapted)

    def _check_numpy_arrays(self, data, name, in_val=""):
        """Check if all elements in the nested structure are numpy arrays."""
        if not all([isinstance(a, np.ndarray) for a in tree.flatten(data)]):
            raise ValueError(
                "Expected "
                "{name}{in_val} to be a numpy array, got {type}".format(
                    name=name,
                    in_val=in_val,
                    type=[type(a) for a in tree.flatten(data)],
                )
            )

    def _check_array_count(self, actual, expected, name, in_val):
        """Check if the number of arrays matches the expected count."""
        if actual != expected:
            raise ValueError(
                "Expected {name}{in_val} to have {expected} arrays, "
                "but got {actual}".format(
                    name=name,
                    in_val=in_val,
                    expected=expected,
                    actual=actual,
                )
            )

    def _check_data_format(self, x, y, validation=False, predict=False):
        """Check if the dataset has the same number of IOs with the model."""
        if validation:
            in_val = " in validation_data"
        else:
            in_val = ""

        self._check_numpy_arrays(x, "x", in_val)
        if y is not None:
            self._check_numpy_arrays(y, "y", in_val)

        self._check_array_count(
            len(tree.flatten(x)), len(self.inputs), "x", in_val
        )
        # When predicting, y is not required.
        if not predict and y is not None:
            self._check_array_count(
                len(tree.flatten(y)), len(self.outputs), "y", in_val
            )

    def _analyze_data(self, dataset):
        input_analysers = [node.get_analyser() for node in self.inputs]
        output_analysers = [head.get_analyser() for head in self._heads]
        analysers = input_analysers + output_analysers
        np_arrays = tree.flatten(dataset)
        for array, analyser in zip(np_arrays, analysers):
            analyser.update(array)

        for analyser in analysers:
            analyser.finalize()

        for hm, analyser in zip(self.inputs + self._heads, analysers):
            hm.config_from_analyser(analyser)

    def _build_hyper_pipeline(self):
        self.tuner.hyper_pipeline = pipeline.HyperPipeline(
            inputs=[node.get_hyper_preprocessors() for node in self.inputs],
            outputs=[head.get_hyper_preprocessors() for head in self._heads],
        )
        self.tuner.hypermodel.hyper_pipeline = self.tuner.hyper_pipeline

    def _check_and_adapt(self, x, y, validation_data):
        # Convert training data.
        self._check_data_format(x, y)
        x = self._adapt(x, self.inputs)
        y = self._adapt(y, self._heads)

        # Convert validation data
        if validation_data:
            self._check_data_format(*validation_data, validation=True)
            x_val, y_val = validation_data
            x_val = self._adapt(x_val, self.inputs)
            y_val = self._adapt(y_val, self._heads)
            validation_data = (x_val, y_val)

        return (x, y), validation_data

    def predict(self, x, batch_size=32, verbose=1, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            batch_size: Number of samples per batch.
                If unspecified, batch_size will default to 32.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
                Controls the verbosity of
                [keras.Model.predict](https://keras.io/api/models/model_training_apis/#predict-method)
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        self._check_data_format(x, None, predict=True)
        dataset = self._adapt(x, self.inputs)
        pipeline = self.tuner.get_best_pipeline()
        model = self.tuner.get_best_model()
        dataset = pipeline.transform_x(dataset)
        y = utils.predict_with_adaptive_batch_size(
            model=model,
            batch_size=batch_size,
            x=dataset,
            verbose=verbose,
            **kwargs
        )
        return pipeline.postprocess(y)

    def evaluate(self, x, y=None, batch_size=32, verbose=1, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            batch_size: Number of samples per batch.
                If unspecified, batch_size will default to 32.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
                Controls the verbosity of
                [keras.Model.evaluate](https://keras.io/api/models/model_training_apis/#evaluate-method)
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs and/or
            metrics). The attribute model.metrics_names will give you the
            display labels for the scalar outputs.
        """
        self._check_data_format(x, y)
        x = self._adapt(x, self.inputs)
        y = self._adapt(y, self._heads)
        pipeline = self.tuner.get_best_pipeline()
        x, y = pipeline.transform((x, y))
        model = self.tuner.get_best_model()
        return utils.evaluate_with_adaptive_batch_size(
            model=model,
            batch_size=batch_size,
            x=x,
            y=y,
            verbose=verbose,
            **kwargs
        )

    def export_model(self):
        """Export the best Keras Model.

        # Returns
            keras.Model instance. The best model found during the search, loaded
            with trained weights.
        """
        return self.tuner.get_best_model()
