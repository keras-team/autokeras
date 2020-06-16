import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras.engine import node as node_module
from autokeras.engine import serializable
from autokeras.utils import utils


class Block(kerastuner.HyperModel, serializable.Serializable):
    """The base class for different Block.

    The Block can be connected together to build the search space
    for an AutoModel. Notably, many args in the __init__ function are defaults to
    be a tunable variable when not specified by the user.

    # Arguments
        name: String. The name of the block. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: str = None, **kwargs):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
            name = utils.to_snake_case(name)
        super().__init__(name=name, **kwargs)
        self.inputs = None
        self.outputs = None
        self._num_output_node = 1

    def _build_wrapper(self, hp, *args, **kwargs):
        with hp.name_scope(self.name):
            return super()._build_wrapper(hp, *args, **kwargs)

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
                raise TypeError('Expect the inputs to block {name} to be '
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
        return {'name': self.name, 'tunable': self.tunable}
