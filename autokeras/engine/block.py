import kerastuner
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.engine import block as block_module
from autokeras.engine import node as node_module
from autokeras.engine import picklable


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
            if not isinstance(input_node, node_module.Node):
                raise TypeError('Expect the inputs to layer {name} to be '
                                'a Node, but got {type}.'.format(
                                    name=self.name,
                                    type=type(input_node)))
            input_node.add_out_block(self)
        self.outputs = []
        for _ in range(self._num_output_node):
            output_node = node_module.Node()
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
            'output_shape': self.output_shape
        })
        return config

    def build(self, hp, inputs=None):
        raise NotImplementedError
