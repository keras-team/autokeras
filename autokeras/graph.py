import queue

import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import hypermodels
from autokeras import nodes
from autokeras.engine import head as head_module
from autokeras.engine import picklable


def embedding_max_features(embedding_block):
    """Fetch the max_features value for embedding block from TextToIntSequence."""
    if embedding_block.max_features:
        return
    input_node = embedding_block.inputs[0]
    while True:
        if not input_node.in_blocks:
            raise ValueError('If Embedding block is not using with '
                             'TextToIntSequence, max_features must be '
                             'specified.')
        block = input_node.in_blocks[0]
        if isinstance(block, hypermodels.TextToIntSequence):
            embedding_block.max_features = block.max_tokens
            return
        input_node = block.inputs[0]


def fetch_heads(source_block):
    """Get the downstream head blocks for a given block in the network.

    # Arguments
        source_block: Block. The source block for the search for heads.

    # Returns
        A list of Head instances.
    """
    heads = []
    visited_blocks = set()
    visited_blocks.add(source_block)
    q = queue.Queue()
    q.put(source_block)
    while not q.empty():
        block = q.get()
        if isinstance(block, head_module.Head):
            heads.append(block)
        for output_node in block.outputs:
            for next_block in output_node.out_blocks:
                if next_block not in visited_blocks:
                    visited_blocks.add(next_block)
                    q.put(next_block)
    return heads


def feature_encoding_input(block):
    """Fetch the column_types and column_names.

    The values are fetched for FeatureEncoding from StructuredDataInput.
    """
    if not isinstance(block.inputs[0], nodes.StructuredDataInput):
        raise TypeError('FeatureEncoding block can only be used '
                        'with StructuredDataInput.')
    block.column_types = block.inputs[0].column_types
    block.column_names = block.inputs[0].column_names


# Compile the graph.
COMPILE_FUNCTIONS = {
    hypermodels.Embedding: [embedding_max_features],
    hypermodels.StructuredDataBlock: [feature_encoding_input],
    hypermodels.FeatureEncoding: [feature_encoding_input],
}

ALL_CLASSES = {
    **vars(nodes),
    **vars(hypermodels),
}


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects={**ALL_CLASSES},
        custom_objects=custom_objects,
        printable_module_name='graph')


class Graph(kerastuner.HyperModel, picklable.Picklable):
    """A graph consists of connected Blocks, or Heads.

    # Arguments
        inputs: A list of input node(s) for the Graph.
        outputs: A list of output node(s) for the Graph.
        override_hps: A list of HyperParameters. The predefined HyperParameters that
            will override the space of the Hyperparameters defined in the Hypermodels
            with the same names.
    """

    def __init__(self, inputs=None, outputs=None, override_hps=None):
        super().__init__()
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self._node_to_id = {}
        self._nodes = []
        self.blocks = []
        self._block_to_id = {}
        if inputs and outputs:
            self._build_network()
        self.override_hps = override_hps or []

    def compile(self):
        """Share the information between blocks."""
        for block in self.blocks:
            for func in COMPILE_FUNCTIONS.get(block.__class__, []):
                func(block)

    def _register_hps(self, hp):
        """Register the override HyperParameters for current HyperParameters."""
        for single_hp in self.override_hps:
            name = single_hp.name
            if name not in hp.values:
                hp.register(single_hp.name,
                            single_hp.__class__.__name__,
                            single_hp.get_config())
                hp.values[name] = single_hp.default

    def _build_network(self):
        self._node_to_id = {}

        # Recursively find all the interested nodes.
        for input_node in self.inputs:
            self._search_network(input_node, self.outputs, set(), set())
        self._nodes = sorted(list(self._node_to_id.keys()),
                             key=lambda x: self._node_to_id[x])

        for node in (self.inputs + self.outputs):
            if node not in self._node_to_id:
                raise ValueError('Inputs and outputs not connected.')

        # Find the blocks.
        blocks = []
        for input_node in self._nodes:
            for block in input_node.out_blocks:
                if any([output_node in self._node_to_id
                        for output_node in block.outputs]) and block not in blocks:
                    blocks.append(block)

        # Check if all the inputs of the blocks are set as inputs.
        for block in blocks:
            for input_node in block.inputs:
                if input_node not in self._node_to_id:
                    raise ValueError('A required input is missing for HyperModel '
                                     '{name}.'.format(name=block.name))

        # Calculate the in degree of all the nodes
        in_degree = [0] * len(self._nodes)
        for node_id, node in enumerate(self._nodes):
            in_degree[node_id] = len([
                block for block in node.in_blocks if block in blocks])

        # Add the blocks in topological order.
        self.blocks = []
        self._block_to_id = {}
        while len(blocks) != 0:
            new_added = []

            # Collect blocks with in degree 0.
            for block in blocks:
                if any([in_degree[self._node_to_id[node]]
                        for node in block.inputs]):
                    continue
                new_added.append(block)

            # Remove the collected blocks from blocks.
            for block in new_added:
                blocks.remove(block)

            for block in new_added:
                # Add the collected blocks to the Graph.
                self._add_block(block)

                # Decrease the in degree of the output nodes.
                for output_node in block.outputs:
                    if output_node not in self._node_to_id:
                        continue
                    output_node_id = self._node_to_id[output_node]
                    in_degree[output_node_id] -= 1

    def _search_network(self, input_node, outputs, in_stack_nodes,
                        visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for block in input_node.out_blocks:
            for output_node in block.outputs:
                if output_node in in_stack_nodes:
                    raise ValueError('The network has a cycle.')
                if output_node not in visited_nodes:
                    self._search_network(output_node, outputs, in_stack_nodes,
                                         visited_nodes)
                if output_node in self._node_to_id.keys():
                    outputs_reached = True

        if outputs_reached:
            self._add_node(input_node)

        in_stack_nodes.remove(input_node)

    def _add_block(self, block):
        if block not in self.blocks:
            block_id = len(self.blocks)
            self._block_to_id[block] = block_id
            self.blocks.append(block)

    def _add_node(self, input_node):
        if input_node not in self._node_to_id:
            self._node_to_id[input_node] = len(self._node_to_id)

    def _get_block(self, name):
        for block in self.blocks:
            if block.name == name:
                return block
        raise ValueError('Cannot find block named {name}.'.format(name=name))

    def get_config(self):
        blocks = [serialize(block) for block in self.blocks]
        nodes = {str(self._node_to_id[node]): serialize(node)
                 for node in self.inputs}
        override_hps = [tf.keras.utils.serialize_keras_object(hp)
                        for hp in self.override_hps]
        block_inputs = {
            str(block_id): [self._node_to_id[node]
                            for node in block.inputs]
            for block_id, block in enumerate(self.blocks)}
        block_outputs = {
            str(block_id): [self._node_to_id[node]
                            for node in block.outputs]
            for block_id, block in enumerate(self.blocks)}

        outputs = [self._node_to_id[node] for node in self.outputs]

        return {
            'override_hps': override_hps,  # List [serialized].
            'blocks': blocks,  # Dict {id: serialized}.
            'nodes': nodes,  # Dict {id: serialized}.
            'outputs': outputs,  # List of node_ids.
            'block_inputs': block_inputs,  # Dict {id: List of node_ids}.
            'block_outputs': block_outputs,  # Dict {id: List of node_ids}.
        }

    @classmethod
    def from_config(cls, config):
        blocks = [deserialize(block) for block in config['blocks']]
        nodes = {int(node_id): deserialize(node)
                 for node_id, node in config['nodes'].items()}
        override_hps = [kerastuner.engine.hyperparameters.deserialize(config)
                        for config in config['override_hps']]

        inputs = [nodes[node_id] for node_id in nodes]
        for block_id, block in enumerate(blocks):
            input_nodes = [nodes[node_id]
                           for node_id in config['block_inputs'][str(block_id)]]
            output_nodes = nest.flatten(block(input_nodes))
            for output_node, node_id in zip(
                    output_nodes, config['block_outputs'][str(block_id)]):
                nodes[node_id] = output_node

        outputs = [nodes[node_id] for node_id in config['outputs']]
        return cls(inputs=inputs, outputs=outputs, override_hps=override_hps)

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        self._register_hps(hp)
        self.compile()
        real_nodes = {}
        for input_node in self.inputs:
            node_id = self._node_to_id[input_node]
            real_nodes[node_id] = input_node.build()
        for block in self.blocks:
            temp_inputs = [real_nodes[self._node_to_id[input_node]]
                           for input_node in block.inputs]
            outputs = block.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(block.outputs, outputs):
                real_nodes[self._node_to_id[output_node]] = real_output_node
        model = tf.keras.Model(
            [real_nodes[self._node_to_id[input_node]] for input_node in
             self.inputs],
            [real_nodes[self._node_to_id[output_node]] for output_node in
             self.outputs])

        return self._compile_keras_model(hp, model)

    def _get_metrics(self):
        metrics = {}
        for output_node in self.outputs:
            block = output_node.in_blocks[0]
            if isinstance(block, head_module.Head):
                metrics[block.name] = block.metrics
        return metrics

    def _get_loss(self):
        loss = {}
        for output_node in self.outputs:
            block = output_node.in_blocks[0]
            if isinstance(block, head_module.Head):
                loss[block.name] = block.loss
        return loss

    def _compile_keras_model(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer = hp.Choice('optimizer',
                              ['adam', 'adadelta', 'sgd'],
                              default='adam')

        model.compile(optimizer=optimizer,
                      metrics=self._get_metrics(),
                      loss=self._get_loss())

        return model
