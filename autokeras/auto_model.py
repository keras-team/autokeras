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

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils

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

    The AutoModel has two use cases. In the first case, the user only specifies the
    input nodes and output heads of the AutoModel. The AutoModel infers the rest part
    of the model. In the second case, user can specify the high-level architecture of
    the AutoModel by connecting the Blocks with the functional API, which is the same
    as the Keras [functional API](https://www.tensorflow.org/guide/keras/functional).

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
        project_name: String. The name of the AutoModel. Defaults to 'auto_model'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        tuner: String or subclass of AutoTuner. If string, it should be one of
            'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass
            of AutoTuner. Defaults to 'greedy'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
        **kwargs: Any arguments supported by kerastuner.Tuner.
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
        **kwargs
    ):
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.seed = seed
        if seed:
            np.random.seed(seed)
            tf.random.set_seed(seed)
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
        inputs = nest.flatten(self.inputs)
        outputs = nest.flatten(self.outputs)

        middle_nodes = []
        for input_node in inputs:
            middle_nodes.append(input_node.get_block()(input_node))

        # Merge the middle nodes.
        if len(middle_nodes) > 1:
            output_node = blocks.Merge()(middle_nodes)
        else:
            output_node = middle_nodes[0]

        outputs = nest.flatten(
            [output_blocks(output_node) for output_blocks in outputs]
        )
        return graph_module.Graph(inputs=inputs, outputs=outputs)

    def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            graph = graph_module.Graph(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, head_module.Head) for output in self.outputs]):
            graph = self._assemble()
            self.outputs = graph.outputs

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
        **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
            batch_size: Int. Number of samples per gradient update. Defaults to 32.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, by default we train for a maximum of 1000 epochs,
                but we stop training if the validation loss stops improving for 10
                epochs (unless you specified an EarlyStopping callback as part of
                the callbacks argument, in which case the EarlyStopping callback you
                specified will determine early stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        self.batch_size = batch_size

        # Check validation information.
        if not validation_data and not validation_split:
            raise ValueError(
                "Either validation_data or a non-zero validation_split "
                "should be provided."
            )

        if validation_data:
            validation_split = 0

        dataset, validation_data = self._convert_to_dataset(
            x=x, y=y, validation_data=validation_data
        )
        self._analyze_data(dataset)
        self._build_hyper_pipeline(dataset)

        # Split the data with validation_split.
        if validation_data is None and validation_split:
            dataset, validation_data = data_utils.split_dataset(
                dataset, validation_split
            )

        self.tuner.search(
            x=dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split,
            **kwargs
        )

    def _adapt(self, dataset, hms):
        if isinstance(dataset, tf.data.Dataset):
            sources = data_utils.unzip_dataset(dataset)
        else:
            sources = nest.flatten(dataset)
        adapted = []
        for source, hm in zip(sources, hms):
            source = hm.get_adapter().adapt(source, self.batch_size)
            adapted.append(source)
        if len(adapted) == 1:
            return adapted[0]
        return tf.data.Dataset.zip(tuple(adapted))

    def _check_data_format(self, dataset, validation=False, predict=False):
        """Check if the dataset has the same number of IOs with the model."""
        if validation:
            in_val = " in validation_data"
            if isinstance(dataset, tf.data.Dataset):
                x = dataset
                y = None
            else:
                x, y = dataset
        else:
            in_val = ""
            x, y = dataset

        if isinstance(x, tf.data.Dataset) and y is not None:
            raise ValueError(
                "Expected y to be None when x is "
                "tf.data.Dataset{in_val}.".format(in_val=in_val)
            )

        if isinstance(x, tf.data.Dataset):
            if not predict:
                x_shapes, y_shapes = data_utils.dataset_shape(x)
                x_shapes = nest.flatten(x_shapes)
                y_shapes = nest.flatten(y_shapes)
            else:
                x_shapes = nest.flatten(data_utils.dataset_shape(x))
        else:
            x_shapes = [a.shape for a in nest.flatten(x)]
            if not predict:
                y_shapes = [a.shape for a in nest.flatten(y)]

        if len(x_shapes) != len(self.inputs):
            raise ValueError(
                "Expected x{in_val} to have {input_num} arrays, "
                "but got {data_num}".format(
                    in_val=in_val, input_num=len(self.inputs), data_num=len(x_shapes)
                )
            )
        if not predict and len(y_shapes) != len(self.outputs):
            raise ValueError(
                "Expected y{in_val} to have {output_num} arrays, "
                "but got {data_num}".format(
                    in_val=in_val,
                    output_num=len(self.outputs),
                    data_num=len(y_shapes),
                )
            )

    def _analyze_data(self, dataset):
        input_analysers = [node.get_analyser() for node in self.inputs]
        output_analysers = [head.get_analyser() for head in self._heads]
        analysers = input_analysers + output_analysers
        for x, y in dataset:
            x = nest.flatten(x)
            y = nest.flatten(y)
            for item, analyser in zip(x + y, analysers):
                analyser.update(item)

        for analyser in analysers:
            analyser.finalize()

        for hm, analyser in zip(self.inputs + self._heads, analysers):
            hm.config_from_analyser(analyser)

    def _build_hyper_pipeline(self, dataset):
        self.tuner.hyper_pipeline = pipeline.HyperPipeline(
            inputs=[node.get_hyper_preprocessors() for node in self.inputs],
            outputs=[head.get_hyper_preprocessors() for head in self._heads],
        )

    def _convert_to_dataset(self, x, y, validation_data):
        """Convert the data to tf.data.Dataset."""
        # TODO: Handle other types of input, zip dataset, tensor, dict.

        # Convert training data.
        self._check_data_format((x, y))
        if isinstance(x, tf.data.Dataset):
            dataset = x
            x = dataset.map(lambda x, y: x)
            y = dataset.map(lambda x, y: y)
        x = self._adapt(x, self.inputs)
        y = self._adapt(y, self._heads)
        dataset = tf.data.Dataset.zip((x, y))

        # Convert validation data
        if validation_data:
            self._check_data_format(validation_data, validation=True)
            if isinstance(validation_data, tf.data.Dataset):
                dataset = validation_data
                x = dataset.map(lambda x, y: x)
                y = dataset.map(lambda x, y: y)
            else:
                x, y = validation_data
            x = self._adapt(x, self.inputs)
            y = self._adapt(y, self._heads)
            validation_data = tf.data.Dataset.zip((x, y))

        return dataset, validation_data

    def _has_y(self, dataset):
        """Remove y from the tf.data.Dataset if exists."""
        shapes = data_utils.dataset_shape(dataset)
        # Only one or less element in the first level.
        if len(shapes) <= 1:
            return False
        # The first level has more than 1 element.
        # The nest has 2 levels.
        for shape in shapes:
            if isinstance(shape, tuple):
                return True
        # The nest has one level.
        # It matches the single IO case.
        if len(shapes) == 2 and len(self.inputs) == 1 and len(self.outputs) == 1:
            return True
        return False

    def predict(self, x, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        if isinstance(x, tf.data.Dataset):
            if self._has_y(x):
                x = x.map(lambda x, y: x)
        self._check_data_format((x, None), predict=True)
        dataset = self._adapt(x, self.inputs)
        pipeline = self.tuner.get_best_pipeline()
        model = self.tuner.get_best_model()
        dataset = pipeline.transform_x(dataset)
        y = model.predict(dataset, **kwargs)
        return pipeline.postprocess(y)

    def evaluate(self, x, y=None, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        self._check_data_format((x, y))
        if isinstance(x, tf.data.Dataset):
            dataset = x
            x = dataset.map(lambda x, y: x)
            y = dataset.map(lambda x, y: y)
        x = self._adapt(x, self.inputs)
        y = self._adapt(y, self._heads)
        dataset = tf.data.Dataset.zip((x, y))
        pipeline = self.tuner.get_best_pipeline()
        dataset = pipeline.transform(dataset)
        return self.tuner.get_best_model().evaluate(x=dataset, **kwargs)

    def export_model(self):
        """Export the best Keras Model.

        # Returns
            tf.keras.Model instance. The best model found during the search, loaded
            with trained weights.
        """
        return self.tuner.get_best_model()
