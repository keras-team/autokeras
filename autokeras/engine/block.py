import kerastuner
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.engine import block as block_module
from autokeras.engine import picklable


class Node(picklable.Picklable):
    """The nodes in a network connecting the blocks."""

    def __init__(self, shape=None):
        super().__init__()
        self.in_blocks = []
        self.out_blocks = []
        self.shape = shape

    def add_in_block(self, hypermodel):
        self.in_blocks.append(hypermodel)

    def add_out_block(self, hypermodel):
        self.out_blocks.append(hypermodel)

    def build(self):
        return tf.keras.Input(shape=self.shape, dtype=tf.float32)

    def get_config(self):
        return {}

    def get_state(self):
        return {'shape': self.shape}

    def set_state(self, state):
        self.shape = state['shape']


class Block(kerastuner.HyperModel, picklable.Picklable):
    """The base class for different Block.

    The Block can be connected together to build the search space
    for an AutoModel. Notably, many args in the __init__ function are defaults to
    be a tunable variable when not specified by the user.

    # Arguments
        name: String. The name of the block. If unspecified, it will be set
        automatically with the class name.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
            name = utils.to_snake_case(name)
        self.name = name
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1

    def _build_wrapper(self, hp, *args, **kwargs):
        if not self.tunable:
            # Copy `HyperParameters` object so that new entries are not added
            # to the search space.
            hp = hp.copy()

        with hp.name_scope(self.name):
            return self._build(hp, *args, **kwargs)

    def __call__(self, inputs):
        """Functional API.

        # Arguments
            inputs: A list of input node(s) or a single input node for the block.

        # Returns
            list: A list of output node(s) of the Block.
        """
        self.inputs = nest.flatten(inputs)
        for input_node in self.inputs:
            if not isinstance(input_node, Node):
                raise TypeError('Expect the inputs to layer {name} to be '
                                'a Node, but got {type}.'.format(
                                    name=self.name,
                                    type=type(input_node)))
            input_node.add_out_block(self)
        self.outputs = []
        for _ in range(self._num_output_node):
            output_node = Node()
            output_node.add_in_block(self)
            self.outputs.append(output_node)
        return self.outputs

    def build(self, hp, inputs=None):
        """Build the Block into a real Keras Model.

        The subclasses should override this function and return the output node.

        # Arguments
            hp: HyperParameters. The hyperparameters for building the model.
            inputs: A list of input node(s).
        """
        return super().build(hp)

    def get_config(self):
        """Get the configuration of the preprocessor.

        # Returns
            A dictionary of configurations of the preprocessor.
        """
        return {'name': self.name}

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


class Head(block_module.Block):
    """Base class for the heads, e.g. classification, regression.

    # Arguments
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be inferred from the AutoModel.
    """

    def __init__(self, loss=None, metrics=None, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        self.loss = loss
        self.metrics = metrics
        # Mark if the head should directly output the input tensor.

    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'metrics': self.metrics,
        })
        return config

    def get_state(self):
        return {'output_shape': self.output_shape}

    def set_state(self, state):
        self.output_shape = state['output_shape']

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def _check(self, y):
        supported_types = (tf.data.Dataset, np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(y, supported_types):
            raise TypeError('Expect the target data of {name} to be tf.data.Dataset,'
                            ' np.ndarray, pd.DataFrame or pd.Series, but got {type}.'
                            .format(name=self.name, type=type(y)))

    def _record_dataset_shape(self, dataset):
        self.output_shape = utils.dataset_shape(dataset)

    def _fit(self, y):
        pass

    def fit_transform(self, y):
        self._check(y)
        self._fit(y)
        dataset = self._convert_to_dataset(y)
        self._record_dataset_shape(dataset)
        return dataset

    def transform(self, y):
        """Transform y into a compatible type (tf.data.Dataset)."""
        self._check(y)
        dataset = self._convert_to_dataset(y)
        return dataset

    def _convert_to_dataset(self, y):
        if isinstance(y, tf.data.Dataset):
            return y
        if isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            return tf.data.Dataset.from_tensor_slices(y)
        if isinstance(y, pd.DataFrame):
            return tf.data.Dataset.from_tensor_slices(y.values)
        if isinstance(y, pd.Series):
            return tf.data.Dataset.from_tensor_slices(y.values.reshape(-1, 1))

    def postprocess(self, y):
        """Postprocess the output of the Keras Model."""
        return y
