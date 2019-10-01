import copy
import functools

import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import head
from autokeras.hypermodel import hyperblock
from autokeras.hypermodel import preprocessor


class GraphHyperModelBase(kerastuner.HyperModel):
    """A HyperModel based on connected Blocks or HyperBlocks.

    # Arguments
        inputs: A list of input node(s) for the GraphHyperModel.
        outputs: A list of output node(s) for the GraphHyperModel.
        name: String. The name of the GraphHyperModel.
    """
    def __init__(self, inputs, outputs, name=None):
        super().__init__(name=name)
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self._node_to_id = {}
        self._nodes = []
        self._blocks = []
        self._block_to_id = {}
        self._build_network()
        self._hps = []
        self.compile()

    def compile(self):
        """Passing config infomation from block to block."""
        for block in self._blocks:
            block.compile()

    def _init_hps(self, hp):
        for single_hp in self._hps:
            name = single_hp.name
            if name not in hp.values:
                hp.space.append(single_hp)
                hp.values[name] = single_hp.default

    def set_hps(self, hps):
        """Set the hyperparameters to constrain the search space.

        # Arguments
            hps: A list of Hyperparameters instances.
        """
        self._hps = hps

    def set_io_shapes(self, dataset):
        """Set the input and output shapes to the nodes.

        Args:
            dataset: tf.data.Dataset. The input dataset before preprocessing.
        """
        # TODO: Set the shapes only if they are not provided by the user when
        #  initiating the HyperHead or Block.
        x_shapes, y_shapes = utils.dataset_shape(dataset)
        for x_shape, input_node in zip(x_shapes, self.inputs):
            input_node.shape = tuple(x_shape.as_list())
        for y_shape, output_node in zip(y_shapes, self.outputs):
            output_node.shape = tuple(y_shape.as_list())
            output_node.in_blocks[0].output_shape = output_node.shape

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
        blocks = set()
        for input_node in self._nodes:
            for block in input_node.out_blocks:
                if any([output_node in self._node_to_id
                        for output_node in block.outputs]):
                    blocks.add(block)

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
        self._blocks = []
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
        if block not in self._blocks:
            block_id = len(self._blocks)
            self._block_to_id[block] = block_id
            self._blocks.append(block)

    def _add_node(self, input_node):
        if input_node not in self._node_to_id:
            self._node_to_id[input_node] = len(self._node_to_id)


class HyperBuiltGraphHyperModel(GraphHyperModelBase):
    """A HyperModel based on connected Blocks.

    It is used by GraphHyperModel. GraphHyperModel's hyper_build function produces
    an instance of HyperBuiltGraphHyperModel, which can be directly built into Keras
    Model.

    # Arguments
        inputs: A list of input node(s) for the HyperBuiltGraphHyperModel.
        outputs: A list of output node(s) for the HyperBuiltGraphHyperModel.
    """

    def __init__(self, inputs, outputs, **kwargs):
        self._model_inputs = []
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def _build_network(self):
        super()._build_network()
        # Find the model input nodes
        for node in self._nodes:
            if self._is_model_inputs(node):
                self._model_inputs.append(node)

        self._model_inputs = sorted(self._model_inputs,
                                    key=lambda x: self._node_to_id[x])

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        self._init_hps(hp)
        real_nodes = {}
        for input_node in self._model_inputs:
            node_id = self._node_to_id[input_node]
            real_nodes[node_id] = input_node.build()
        for block in self._blocks:
            if isinstance(block, preprocessor.Preprocessor):
                continue
            temp_inputs = [real_nodes[self._node_to_id[input_node]]
                           for input_node in block.inputs]
            outputs = block.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(block.outputs, outputs):
                real_nodes[self._node_to_id[output_node]] = real_output_node
        model = tf.keras.Model(
            [real_nodes[self._node_to_id[input_node]] for input_node in
             self._model_inputs],
            [real_nodes[self._node_to_id[output_node]] for output_node in
             self.outputs])

        return self._compile_keras_model(hp, model)

    def _get_metrics(self):
        metrics = {}
        for output_node in self.outputs:
            block = output_node.in_blocks[0]
            if isinstance(block, head.Head):
                metrics[block.name] = block.metrics
        return metrics

    def _get_loss(self):
        loss = {}
        for output_node in self.outputs:
            block = output_node.in_blocks[0]
            if isinstance(block, head.Head):
                loss[block.name] = block.loss
        return loss

    def _compile_keras_model(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer = hp.Choice('optimizer',
                              ['adam',
                               'adadelta',
                               'sgd'])

        model.compile(optimizer=optimizer,
                      metrics=self._get_metrics(),
                      loss=self._get_loss())

        return model

    def preprocess(self, hp, dataset, validation_data=None, fit=False):
        for block in self._blocks:
            if isinstance(block, preprocessor.Preprocessor):
                block.set_hp(hp)
        dataset = self._preprocess(dataset, fit=fit)
        if not validation_data:
            return dataset
        validation_data = self._preprocess(validation_data)
        return dataset, validation_data

    def _preprocess(self, dataset, fit=False):
        # A list of input node ids in the same order as the x in the dataset.
        input_node_ids = [self._node_to_id[input_node] for input_node in self.inputs]

        # Iterate until all the model inputs have their data.
        while set(map(lambda node: self._node_to_id[node], self._model_inputs)
                  ) - set(input_node_ids):
            # Gather the blocks for the next iteration over the dataset.
            blocks = []
            for node_id in input_node_ids:
                for block in self._nodes[node_id].out_blocks:
                    if isinstance(block, preprocessor.Preprocessor):
                        blocks.append(block)
            if fit:
                # Iterate the dataset to fit the preprocessors in current depth.
                self._fit_preprocessors(dataset, input_node_ids, blocks)

            # Transform the dataset.
            output_node_ids = []
            dataset = dataset.map(functools.partial(
                self._preprocess_transform,
                input_node_ids=input_node_ids,
                output_node_ids=output_node_ids,
                blocks=blocks,
                fit=fit))

            # Build input_node_ids for next depth.
            input_node_ids = output_node_ids
        return dataset

    def _fit_preprocessors(self, dataset, input_node_ids, blocks):
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

    def _preprocess_transform(self,
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
            if self._is_model_inputs(self._nodes[node_id]):
                output_data[node_id] = data

        for node_id in sorted(output_data.keys()):
            output_node_ids.append(node_id)
        return tuple(map(
            lambda node_id: output_data[node_id], output_node_ids)), y

    @staticmethod
    def _is_model_inputs(node):
        for block in node.in_blocks:
            if not isinstance(block, preprocessor.Preprocessor):
                return False
        for block in node.out_blocks:
            if not isinstance(block, preprocessor.Preprocessor):
                return True
        return False

    def save_preprocessors(self, path):
        configs = {}
        weights = {}
        for block in self._blocks:
            if isinstance(block, preprocessor.Preprocessor):
                configs[block.name] = block.get_config()
                weights[block.name] = block.get_weights()
        preprocessors = {'configs': configs, 'weights': weights}
        utils.pickle_to_file(preprocessors, path)

    def load_preprocessors(self, path):
        preprocessors = utils.pickle_from_file(path)
        configs = preprocessors['configs']
        weights = preprocessors['weights']
        for name, config in configs.items():
            block = self._get_block(name)
            block.set_config(config)
        for name, weight in weights.items():
            block = self._get_block(name)
            block.set_weights(weight)

    def clear_preprocessors(self):
        for block in self._blocks:
            if isinstance(block, preprocessor.Preprocessor):
                block.clear_weights()

    def _get_block(self, name):
        for block in self._blocks:
            if block.name == name:
                return block
        return None


def copy_block(old_block):
    # TODO: use get_config and set_config, which requires the implementation of
    # these two functions in all blocks.
    block = copy.copy(old_block)
    block.clear_nodes()
    if isinstance(block, preprocessor.Preprocessor):
        block.clear_weights()
    return block


class GraphHyperModel(GraphHyperModelBase):
    """A HyperModel based on connected Blocks and HyperBlocks.

    # Arguments
        inputs: A list of input node(s) for the GraphHyperModel.
        outputs: A list of output node(s) for the GraphHyperModel.
    """

    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.hyper_built_ghm = None

    def build(self, hp):
        """Build the HyperModel into a Keras Model."""
        return self.hyper_built_ghm.build(hp)

    def hyper_build(self, hp):
        """Build a GraphHyperModel with no HyperBlock but only Block."""
        inputs = copy.copy(self.inputs)
        old_node_to_new = {}
        for input_node, old_input_node in zip(inputs, self.inputs):
            input_node.clear_edges()
            old_node_to_new[old_input_node] = input_node
        for old_block in self._blocks:
            block = copy_block(old_block)
            inputs = [old_node_to_new[input_node]
                      for input_node in old_block.inputs]
            if isinstance(old_block, hyperblock.HyperBlock):
                outputs = block.build(hp, inputs=inputs)
            else:
                outputs = block(inputs)
            for output_node, old_output_node in zip(outputs, old_block.outputs):
                old_node_to_new[old_output_node] = output_node
        inputs = []
        for input_node in self.inputs:
            inputs.append(old_node_to_new[input_node])
        outputs = []
        for output_node in self.outputs:
            outputs.append(old_node_to_new[output_node])
        self.hyper_built_ghm = HyperBuiltGraphHyperModel(inputs, outputs)
        self.hyper_built_ghm.set_hps(self._hps)

    def preprocess(self, hp, dataset, validation_data=None, fit=False):
        """Preprocess the data to be ready for the Keras Model.

        # Arguments
            hp: HyperParameters. Used to build the HyperModel.
            dataset: tf.data.Dataset. Training data.
            validation_data: tf.data.Dataset. Validation data.
            fit: Boolean. Whether to fit the preprocessing layers with x and y.

        # Returns
            if validation data is provided.
            A tuple of two preprocessed tf.data.Dataset, (train, validation).
            Otherwise, return the training dataset.
        """
        return self.hyper_built_ghm.preprocess(hp, dataset, validation_data, fit)

    def save_preprocessors(self, path):
        """Save the preprocessors in the hypermodel in a single file.

        Args:
            path: String. The path to a single file.
        """
        self.hyper_built_ghm.save_preprocessors(path)

    def load_preprocessors(self, path):
        """Load the preprocessors in the hypermodel from a single file

        Args:
            path: String. The path to a single file.
        """
        self.hyper_built_ghm.load_preprocessors(path)

    def clear_preprocessors(self):
        """Clear the preprocessors' weights in the hypermodel."""
        self.hyper_built_ghm.clear_preprocessors()
