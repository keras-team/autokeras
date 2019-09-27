import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util import nest

import kerastuner
from autokeras import meta_model
from autokeras import tuner
from autokeras import utils
from autokeras.hypermodel import graph
from autokeras.hypermodel import head
from autokeras.hypermodel import node


class AutoModel(object):
    """ A Model defined by inputs and outputs.

    AutoModel combines a HyperModel and a Tuner to tune the HyperModel.
    The user can use it in a similar way to a Keras model since it
    also has `fit()` and  `predict()` methods.

    The user can specify the inputs and outputs of the AutoModel. It will infer
    the rest of the high-level neural architecture.

    # Arguments
        inputs: A list of or a HyperNode instance.
            The input node(s) of the AutoModel.
        outputs: A list of or a HyperHead instance.
            The output head(s) of the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'auto_model'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 name='auto_model',
                 max_trials=100,
                 directory=None,
                 seed=None):
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.name = name
        self.tuner = None
        self.max_trials = max_trials
        self.directory = directory
        self.seed = seed
        self.hypermodel = None
        if all([isinstance(output_node, head.Head)
                for output_node in self.outputs]):
            self.heads = self.outputs
        else:
            self.heads = [output_node.in_blocks[0] for output_node in self.outputs]

    def _meta_build(self, dataset):
        self.hypermodel = meta_model.assemble(inputs=self.inputs,
                                              outputs=self.outputs,
                                              dataset=dataset,
                                              seed=self.seed)
        self.outputs = self.hypermodel.outputs

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, we would use epochs equal to 1000 and early stopping
                with patience equal to 30.
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - tuple `(x_val, y_val)` of Numpy arrays or tensors
                  - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                  - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        dataset, validation_data = self._prepare_data(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split)

        # Initialize the hypermodel.
        self._meta_build(dataset)
        self.hypermodel.set_io_shapes(dataset)

        # Build the hypermodel in tuner init.
        hp = kerastuner.HyperParameters()
        self.hypermodel.hyper_build(hp)
        self.hypermodel.preprocess(
            hp=kerastuner.HyperParameters(),
            dataset=dataset,
            validation_data=validation_data,
            fit=True)
        self.tuner = tuner.RandomSearch(
            hypermodel=self.hypermodel,
            objective='val_loss',
            max_trials=self.max_trials,
            directory=self.directory,
            seed=self.seed,
            project_name=self.name)
        self.hypermodel.clear_preprocessors()

        self.tuner.search(x=dataset,
                          epochs=epochs,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          **kwargs)

    def _process_xy(self, x, y=None, fit=False, predict=False):
        """Convert x, y to tf.data.Dataset.

        # Arguments
            x: Any type allowed by the corresponding input node.
            y: Any type allowed by the corresponding head.
            fit: Boolean. Whether to fit the type converter with the provided data.
            predict: Boolean. If it is called by the predict function of AutoModel.

        # Returns
            A tf.data.Dataset containing both x and y.
        """
        if isinstance(x, tf.data.Dataset):
            if y is None and not predict:
                return x
            if isinstance(y, tf.data.Dataset):
                return tf.data.Dataset.zip((x, y))

        x = nest.flatten(x)
        new_x = []
        for data, input_node in zip(x, self.inputs):
            if fit:
                input_node.fit(data)
            data = input_node.transform(data)
            new_x.append(data)
        x = tf.data.Dataset.zip(tuple(new_x))

        if predict:
            return tf.data.Dataset.zip((x, x))

        if not isinstance(y, tf.data.Dataset):
            y = nest.flatten(y)
            new_y = []
            for data, head_block in zip(y, self.heads):
                if fit:
                    head_block.fit(data)
                data = head_block.transform(data)
                new_y.append(data)
            y = tf.data.Dataset.zip(tuple(new_y))

        return tf.data.Dataset.zip((x, y))

    def _prepare_data(self, x, y, validation_data, validation_split):
        """Convert the data to tf.data.Dataset."""
        # Check validation information.
        if not validation_data and not validation_split:
            raise ValueError('Either validation_data or validation_split'
                             'should be provided.')
        # TODO: Handle other types of input, zip dataset, tensor, dict.
        # Prepare the dataset.
        dataset = self._process_xy(x, y, fit=True)
        if validation_data:
            val_x, val_y = validation_data
            validation_data = self._process_xy(val_x, val_y)
        # Split the data with validation_split.
        if validation_data is None and validation_split:
            dataset, validation_data = utils.split_dataset(dataset, validation_split)
        return dataset, validation_data

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        best_model, x = self._prepare_best_model_and_data(
            x=x,
            y=None,
            batch_size=batch_size,
            predict=True)
        y = best_model.predict(x, **kwargs)
        y = self._postprocess(y)
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        return y

    def _postprocess(self, y):
        y = nest.flatten(y)
        new_y = []
        for temp_y, head_block in zip(y, self.heads):
            if isinstance(head_block, head.Head):
                temp_y = head_block.postprocess(temp_y)
            new_y.append(temp_y)
        return new_y

    def evaluate(self, x, y=None, batch_size=32, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        best_model, data = self._prepare_best_model_and_data(
            x=x,
            y=y,
            batch_size=batch_size)
        return best_model.evaluate(data, **kwargs)

    def _prepare_best_model_and_data(self,
                                     x,
                                     y=None,
                                     batch_size=32,
                                     predict=False):
        best_model = self.tuner.get_best_models(1)[0]
        best_trial = self.tuner.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters

        self.tuner.load_trial(best_trial)
        x = self._process_xy(x, y, predict=predict)
        x = self.hypermodel.preprocess(best_hp, x)
        x = x.batch(batch_size)
        return best_model, x


class GraphAutoModel(AutoModel):
    """A HyperModel defined by a graph of HyperBlocks.

    GraphAutoModel is a subclass of HyperModel. Besides the HyperModel properties,
    it also has a tuner to tune the HyperModel. The user can use it in a similar
    way to a Keras model since it also has `fit()` and  `predict()` methods.

    The user can specify the high-level neural architecture by connecting the
    HyperBlocks with the functional API, which is the same as
    the Keras functional API.

    # Arguments
        inputs: A list of or a HyperNode instances.
            The input node(s) of the GraphAutoModel.
        outputs: A list of or a HyperNode instances.
            The output node(s) of the GraphAutoModel.
        name: String. The name of the AutoModel. Defaults to 'graph_auto_model'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 name='graph_auto_model',
                 max_trials=100,
                 directory=None,
                 seed=None):
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name,
            max_trials=max_trials,
            directory=directory,
            seed=seed
        )
        self.hypermodel = graph.GraphHyperModel(self.inputs, self.outputs)

    def _meta_build(self, dataset):
        pass
