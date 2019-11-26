import functools

import kerastuner
import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module
from tensorflow.python.util import nest

import autokeras
from autokeras.hypermodel import base
from autokeras.hypermodel import compiler


class Graph(base.Weighted, kerastuner.engine.stateful.Stateful):
    """A graph consists of connected Blocks, HyperBlocks, Preprocessors or Heads.

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

    def compile(self, func):
        """Share the information between blocks by calling functions in compiler.

        # Arguments
            func: A dictionary. The keys are the block classes. The values are
                corresponding compile functions.
        """
        for block in self.blocks:
            if block.__class__ in func:
                func[block.__class__](block)

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
                # Add the collected blocks to the AutoModel.
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

    def get_state(self):
        block_states = {str(block_id): block.get_state()
                        for block_id, block in enumerate(self.blocks)}
        block_classes = [block.__class__.__name__ for block in self.blocks]

        node_states = {str(node_id): node.get_state()
                       for node_id, node in enumerate(self._nodes)}
        node_classes = [node.__class__.__name__ for node in self._nodes]

        block_inputs = {
            block_id: [self._node_to_id[node]
                       for node in block.inputs]
            for block_id, block in enumerate(self.blocks)}
        block_outputs = {
            block_id: [self._node_to_id[node]
                       for node in block.outputs]
            for block_id, block in enumerate(self.blocks)}

        node_inputs = {
            node_id: [self._block_to_id[block]
                      for block in node.in_blocks]
            for node_id, node in enumerate(self._nodes)}
        node_outputs = {
            node_id: [self._block_to_id[block]
                      for block in node.out_blocks]
            for node_id, node in enumerate(self._nodes)}

        override_hps = [(hp.__class__.__name__, hp.get_config())
                        for hp in self.override_hps]

        inputs = [self._node_to_id[node] for node in self.inputs]
        outputs = [self._node_to_id[node] for node in self.outputs]

        return {
            'inputs': inputs,  # List of node_ids.
            'outputs': outputs,  # List of node_ids.
            'block_states': block_states,  # Dict {id: state}.
            'block_classes': block_classes,  # List of strings of class names.
            'node_states': node_states,  # Dict {id: state}.
            'node_classes': node_classes,  # List of strings of class names.
            'block_inputs': block_inputs,  # Dict {id: List of node_ids}.
            'block_outputs': block_outputs,  # Dict {id: List of node_ids}.
            'node_inputs': node_inputs,  # Dict {id: List of block_ids}.
            'node_outputs': node_outputs,  # Dict {id: List of block_ids}.
            'override_hps': override_hps,  # List of tuple of (class_name, config).
        }

    def set_state(self, state):
        block_states = state['block_states']
        node_states = state['node_states']

        self.blocks = [getattr(autokeras, block_class)()
                       for block_class in state['block_classes']]
        self._nodes = [getattr(autokeras, node_class)()
                       for node_class in state['node_classes']]

        self._block_to_id = {self.blocks[block_id]: block_id
                             for block_id in range(len(self.blocks))}
        self._node_to_id = {self._nodes[node_id]: node_id
                            for node_id in range(len(self._nodes))}

        for block_id, block in enumerate(self.blocks):
            block.set_state(block_states[str(block_id)])
            block.inputs = [self._nodes[node_id]
                            for node_id in state['block_inputs'][str(block_id)]]
            block.outputs = [self._nodes[node_id]
                             for node_id in state['block_outputs'][str(block_id)]]
        for node_id, node in enumerate(self._nodes):
            node.set_state(node_states[str(node_id)])
            node.in_blocks = [self.blocks[block_id]
                              for block_id in state['node_inputs'][str(node_id)]]
            node.out_blocks = [self.blocks[block_id]
                               for block_id in state['node_outputs'][str(node_id)]]

        self.override_hps = [getattr(hp_module, hp_class).from_config(config)
                             for hp_class, config in state['override_hps']]

        self.inputs = [self._nodes[node_id] for node_id in state['inputs']]
        self.outputs = [self._nodes[node_id] for node_id in state['outputs']]

    def build(self, hp):
        self._register_hps(hp)

    def get_weights(self):
        node_weights = {str(node_id): node.get_weights()
                        for node_id, node in enumerate(self._nodes)}
        return {'nodes': node_weights}

    def set_weights(self, weights):
        node_weights = weights['nodes']
        for node_id, node in enumerate(self._nodes):
            node.set_weights(node_weights[str(node_id)])


class PlainGraph(Graph):
    """A graph built from a HyperGraph to produce KerasGraph and PreprocessGraph.

    A PlainGraph does not contain HyperBlock. HyperGraph's hyper_build function
    returns an instance of PlainGraph, which can be directly built into a KerasGraph
    and a PreprocessGraph.
    """

    def __init__(self, **kwargs):
        self._keras_model_inputs = []
        super().__init__(**kwargs)

    def _build_network(self):
        super()._build_network()
        # Find the model input nodes
        for node in self._nodes:
            if self._is_keras_model_inputs(node):
                self._keras_model_inputs.append(node)

        self._keras_model_inputs = sorted(self._keras_model_inputs,
                                          key=lambda x: self._node_to_id[x])

    @staticmethod
    def _is_keras_model_inputs(node):
        for block in node.in_blocks:
            if not isinstance(block, base.Preprocessor):
                return False
        for block in node.out_blocks:
            if not isinstance(block, base.Preprocessor):
                return True
        return False

    def build_keras_graph(self):
        return KerasGraph(self._keras_model_inputs,
                          self.outputs,
                          override_hps=self.override_hps)

    def build_preprocess_graph(self):
        return PreprocessGraph(self.inputs,
                               self._keras_model_inputs,
                               override_hps=self.override_hps)


class KerasGraph(Graph, kerastuner.HyperModel):
    """A graph and HyperModel to be built into a Keras model."""

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        super().build(hp)
        self.compile(compiler.AFTER)
        real_nodes = {}
        for input_node in self.inputs:
            node_id = self._node_to_id[input_node]
            real_nodes[node_id] = input_node.build()
        for block in self.blocks:
            if isinstance(block, base.Preprocessor):
                continue
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
            if isinstance(block, base.Head):
                metrics[block.name] = block.metrics
        return metrics

    def _get_loss(self):
        loss = {}
        for output_node in self.outputs:
            block = output_node.in_blocks[0]
            if isinstance(block, base.Head):
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


class PreprocessGraph(Graph):
    """A graph consists of only Preprocessors.

    It is both a search space with Hyperparameters and a model to be fitted. It
    preprocess the dataset with the Preprocessors. The output is the input to the
    Keras model. It does not extend Hypermodel class because it cannot be built into
    a Keras model.
    """

    def preprocess(self, dataset, validation_data=None, fit=False):
        """Preprocess the data to be ready for the Keras Model.

        # Arguments
            dataset: tf.data.Dataset. Training data.
            validation_data: tf.data.Dataset. Validation data.
            fit: Boolean. Whether to fit the preprocessing layers with x and y.

        # Returns
            if validation data is provided.
            A tuple of two preprocessed tf.data.Dataset, (train, validation).
            Otherwise, return the training dataset.
        """
        dataset = self._preprocess(dataset, fit=fit)
        if validation_data:
            validation_data = self._preprocess(validation_data)
        return dataset, validation_data

    def _preprocess(self, dataset, fit=False):
        # A list of input node ids in the same order as the x in the dataset.
        input_node_ids = [self._node_to_id[input_node] for input_node in self.inputs]

        # Iterate until all the model inputs have their data.
        while set(map(lambda node: self._node_to_id[node], self.outputs)
                  ) - set(input_node_ids):
            # Gather the blocks for the next iteration over the dataset.
            blocks = []
            for node_id in input_node_ids:
                for block in self._nodes[node_id].out_blocks:
                    if block in self.blocks:
                        blocks.append(block)
            if fit:
                # Iterate the dataset to fit the preprocessors in current depth.
                self._fit(dataset, input_node_ids, blocks)

            # Transform the dataset.
            output_node_ids = []
            dataset = dataset.map(functools.partial(
                self._transform,
                input_node_ids=input_node_ids,
                output_node_ids=output_node_ids,
                blocks=blocks,
                fit=fit))

            # Build input_node_ids for next depth.
            input_node_ids = output_node_ids
        return dataset

    def _fit(self, dataset, input_node_ids, blocks):
        # Iterate the dataset to fit the preprocessors in current depth.
        for x, y in dataset:
            x = nest.flatten(x)
            id_to_data = {
                node_id: temp_x for temp_x, node_id in zip(x, input_node_ids)
            }
            for block in blocks:
                data = [id_to_data[self._node_to_id[input_node]]
                        for input_node in block.inputs]
                block.update(data, y=y)

        # Finalize and set the shapes of the output nodes.
        for block in blocks:
            block.finalize()
            nest.flatten(block.outputs)[0].shape = block.output_shape

    def _transform(self,
                   x,
                   y,
                   input_node_ids,
                   output_node_ids,
                   blocks,
                   fit=False):
        x = nest.flatten(x)
        id_to_data = {
            node_id: temp_x
            for temp_x, node_id in zip(x, input_node_ids)
        }
        output_data = {}
        # Transform each x by the corresponding block.
        for hm in blocks:
            data = [id_to_data[self._node_to_id[input_node]]
                    for input_node in hm.inputs]
            data = tf.py_function(functools.partial(hm.transform, fit=fit),
                                  inp=nest.flatten(data),
                                  Tout=hm.output_types())
            data = nest.flatten(data)[0]
            data.set_shape(hm.output_shape)
            output_data[self._node_to_id[hm.outputs[0]]] = data
        # Keep the Keras Model inputs even they are not inputs to the blocks.
        for node_id, data in id_to_data.items():
            if self._nodes[node_id] in self.outputs:
                output_data[node_id] = data

        for node_id in sorted(output_data.keys()):
            output_node_ids.append(node_id)
        return tuple(map(
            lambda node_id: output_data[node_id], output_node_ids)), y

    def build(self, hp):
        """Obtain the values of all the HyperParameters.

        Different from the build function of Hypermodel. This build function does not
        produce a Keras model. It only obtain the hyperparameter values from
        HyperParameters.

        # Arguments
            hp: HyperParameters.
        """
        super().build(hp)
        self.compile(compiler.BEFORE)
        for block in self.blocks:
            block.build(hp)

    def get_weights(self):
        weights = super().get_weights()
        block_weights = {str(block_id): block.get_weights()
                         for block_id, block in enumerate(self.blocks)}
        weights.update({'blocks': block_weights})
        return weights

    def set_weights(self, weights):
        super().set_weights(weights)
        block_weights = weights['blocks']
        for block_id, block in enumerate(self.blocks):
            block.set_weights(block_weights[str(block_id)])


def copy(old_instance):
    instance = old_instance.__class__()
    instance.set_state(old_instance.get_state())
    if isinstance(instance, base.Weighted):
        instance.set_weights(old_instance.get_weights())
    return instance


class HyperGraph(Graph):
    """A HyperModel based on connected Blocks and HyperBlocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compile(compiler.HYPER)

    def build_graphs(self, hp):
        plain_graph = self.hyper_build(hp)
        preprocess_graph = plain_graph.build_preprocess_graph()
        preprocess_graph.build(hp)
        return (preprocess_graph,
                plain_graph.build_keras_graph())

    def hyper_build(self, hp):
        """Build a GraphHyperModel with no HyperBlock but only Block."""
        # Make sure get_uid would count from start.
        tf.keras.backend.clear_session()
        inputs = []
        old_node_to_new = {}
        for old_input_node in self.inputs:
            input_node = copy(old_input_node)
            inputs.append(input_node)
            old_node_to_new[old_input_node] = input_node
        for old_block in self.blocks:
            inputs = [old_node_to_new[input_node]
                      for input_node in old_block.inputs]
            if isinstance(old_block, base.HyperBlock):
                outputs = old_block.build(hp, inputs=inputs)
            else:
                outputs = copy(old_block)(inputs)
            for output_node, old_output_node in zip(outputs, old_block.outputs):
                old_node_to_new[old_output_node] = output_node
        inputs = []
        for input_node in self.inputs:
            inputs.append(old_node_to_new[input_node])
        outputs = []
        for output_node in self.outputs:
            outputs.append(old_node_to_new[output_node])
        return PlainGraph(inputs, outputs, override_hps=self.override_hps)
