import os

import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import autokeras.utils
from autokeras import meta_model
from autokeras import tuner
from autokeras import utils
from autokeras.hypermodel import graph
from autokeras.hypermodel import head
from autokeras.hypermodel import node
from autokeras.hypermodel import preprocessor


class AutoModel(object):
    """ A Model defined by inputs and outputs.

    AutoModel has a HyperModel and a Tuner to tune the HyperModel.
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
        self._label_encoders = None

    def _meta_build(self, dataset):
        self.hypermodel = meta_model.assemble(inputs=self.inputs,
                                              outputs=self.outputs,
                                              dataset=dataset)
        self.outputs = self.hypermodel.outputs

    def fit(self,
            x=None,
            y=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
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
        dataset, validation_data = self.prepare_data(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split)
        self._meta_build(dataset)
        self.hypermodel.set_io_shapes(dataset)
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

        # TODO: allow early stop if epochs is not specified.
        self.tuner.search(x=dataset,
                          validation_data=validation_data,
                          **kwargs)

    def prepare_data(self, x, y, validation_data, validation_split):
        # Initialize HyperGraph model
        x = nest.flatten(x)
        y = nest.flatten(y)
        # TODO: check x, y types to be numpy.ndarray or tf.data.Dataset.
        # TODO: y.reshape(-1, 1) if needed.
        y = self._label_encoding(y)
        # Split the data with validation_split
        if (all([isinstance(temp_x, np.ndarray) for temp_x in x]) and
                all([isinstance(temp_y, np.ndarray) for temp_y in y]) and
                validation_data is None and
                validation_split):
            (x, y), (x_val, y_val) = utils.split_train_to_valid(
                x, y,
                validation_split)
            validation_data = x_val, y_val
        # TODO: Handle other types of input, zip dataset, tensor, dict.
        # Prepare the dataset
        dataset = x if isinstance(x, tf.data.Dataset) \
            else utils.prepare_preprocess(x, y)
        if not isinstance(validation_data, tf.data.Dataset):
            x_val, y_val = validation_data
            validation_data = utils.prepare_preprocess(x_val, y_val)
        return dataset, validation_data

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: tf.data.Dataset or numpy.ndarray. Testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.
        """
        best_model = self.tuner.get_best_models(1)[0]
        best_trial = self.tuner.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters

        self.tuner.load_trial(best_trial)
        x = utils.prepare_preprocess(x, x)
        x = self.hypermodel.preprocess(best_hp, x)
        x = x.batch(batch_size)
        y = best_model.predict(x, **kwargs)
        y = nest.flatten(y)
        y = self._postprocess(y)
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        return y

    def _label_encoding(self, y):
        self._label_encoders = []
        new_y = []
        for temp_y, output_node in zip(y, self.outputs):
            hyper_head = output_node
            if isinstance(hyper_head, node.Node):
                hyper_head = output_node.in_blocks[0]
            if (isinstance(hyper_head, head.ClassificationHead) and
                    utils.is_label(temp_y)):
                label_encoder = utils.OneHotEncoder()
                label_encoder.fit(y)
                new_y.append(label_encoder.transform(y))
                self._label_encoders.append(label_encoder)
            else:
                new_y.append(temp_y)
                self._label_encoders.append(None)
        return new_y

    def _postprocess(self, y):
        if not self._label_encoders:
            return y
        new_y = []
        for temp_y, label_encoder in zip(y, self._label_encoders):
            if label_encoder:
                new_y.append(label_encoder.inverse_transform(temp_y))
            else:
                new_y.append(temp_y)
        return new_y


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
