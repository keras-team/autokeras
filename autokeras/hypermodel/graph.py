import copy
import functools

import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import head
from autokeras.hypermodel import processor
from autokeras.hypermodel import composite


class GraphHyperModel(kerastuner.HyperModel):
    def __init__(self, inputs, outputs, name=None):
        super().__init__(name=name)
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self._node_to_id = {}
        self._nodes = []
        self._hypermodels = []
        self._hypermodel_to_id = {}
        self._model_inputs = []
        self._hypermodel_topo_depth = {}
        self._total_topo_depth = 0
        self._build_network()
        self._plain_graph_hm = None

    def hyper_build(self, hp):
        if not any([isinstance(hypermodel, composite.CompositeHyperBlock)
                    for hypermodel in self._hypermodels]):
            return
        inputs = copy.copy(self.inputs)
        old_node_to_new = {}
        for input_node, old_input_node in zip(inputs, self.inputs):
            input_node.clear_edges()
            old_node_to_new[old_input_node] = input_node
        for old_hypermodel in self._hypermodels:
            hypermodel = copy.copy(old_hypermodel)
            hypermodel.clear_nodes()
            inputs = [old_node_to_new[input_node]
                      for input_node in old_hypermodel.inputs]
            if isinstance(old_hypermodel, composite.CompositeHyperBlock):
                outputs = hypermodel.build(hp, inputs=inputs)
            else:
                outputs = hypermodel(inputs)
            for output_node, old_output_node in zip(outputs,
                                                    old_hypermodel.outputs):
                old_node_to_new[old_output_node] = output_node
        inputs = []
        for input_node in self.inputs:
            inputs.append(old_node_to_new[input_node])
        outputs = []
        for output_node in self.outputs:
            outputs.append(old_node_to_new[output_node])
        self._plain_graph_hm = GraphHyperModel(inputs, outputs)

    def build(self, hp):
        if any([isinstance(hypermodel, composite.CompositeHyperBlock)
                for hypermodel in self._hypermodels]):
            return self._plain_graph_hm.build(hp)
        real_nodes = {}
        for input_node in self._model_inputs:
            node_id = self._node_to_id[input_node]
            real_nodes[node_id] = input_node.build()
        for hypermodel in self._hypermodels:
            if isinstance(hypermodel, processor.HyperPreprocessor):
                continue
            temp_inputs = [real_nodes[self._node_to_id[input_node]]
                           for input_node in hypermodel.inputs]
            outputs = hypermodel.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(hypermodel.outputs, outputs):
                real_nodes[self._node_to_id[output_node]] = real_output_node
        model = tf.keras.Model(
            [real_nodes[self._node_to_id[input_node]] for input_node in
             self._model_inputs],
            [real_nodes[self._node_to_id[output_node]] for output_node in
             self.outputs])

        return self._compiled(hp, model)

    def set_io_shapes(self, dataset):
        # TODO: Set the shapes only if they are not provided by the user when
        #  initiating the HyperHead or Block.
        x_shapes, y_shapes = utils.dataset_shape(dataset)
        for x_shape, input_node in zip(x_shapes, self.inputs):
            input_node.shape = tuple(x_shape.as_list())
        for y_shape, output_node in zip(y_shapes, self.outputs):
            output_node.shape = tuple(y_shape.as_list())
            output_node.in_hypermodels[0].output_shape = output_node.shape

    def set_node_shapes(self, dataset):
        x_shapes, y_shapes = utils.dataset_shape(dataset)
        for x_shape, input_node in zip(x_shapes, self._model_inputs):
            input_node.shape = tuple(x_shape.as_list())
        for y_shape, output_node in zip(y_shapes, self.outputs):
            output_node.shape = tuple(y_shape.as_list())
            output_node.in_hypermodels[0].output_shape = output_node.shape

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

        # Find the hypermodels.
        hypermodels = set()
        for input_node in self._nodes:
            for hypermodel in input_node.out_hypermodels:
                if any([output_node in self._node_to_id
                        for output_node in hypermodel.outputs]):
                    hypermodels.add(hypermodel)

        # Check if all the inputs of the hypermodels are set as inputs.
        for hypermodel in hypermodels:
            for input_node in hypermodel.inputs:
                if input_node not in self._node_to_id:
                    raise ValueError('A required input is missing for HyperModel '
                                     '{name}.'.format(name=hypermodel.name))

        # Calculate the in degree of all the nodes
        in_degree = [0] * len(self._nodes)
        for node_id, node in enumerate(self._nodes):
            in_degree[node_id] = len([
                hypermodel
                for hypermodel in node.in_hypermodels if hypermodel in hypermodels
            ])

        # Add the hypermodels in topological order.
        self._hypermodels = []
        self._hypermodel_to_id = {}
        self._hypermodel_topo_depth = {}
        depth_count = 0
        while len(hypermodels) != 0:
            new_added = []

            # Collect hypermodels with in degree 0.
            for hypermodel in hypermodels:
                if any([in_degree[self._node_to_id[node]]
                        for node in hypermodel.inputs]):
                    continue
                new_added.append(hypermodel)

            # Remove the collected hypermodels from hypermodels.
            for hypermodel in new_added:
                hypermodels.remove(hypermodel)

            for hypermodel in new_added:
                # Add the collected hypermodels to the AutoModel.
                self._add_hypermodel(hypermodel)
                self._hypermodel_topo_depth[
                    self._hypermodel_to_id[hypermodel]] = depth_count

                # Decrease the in degree of the output nodes.
                for output_node in hypermodel.outputs:
                    if output_node not in self._node_to_id:
                        continue
                    output_node_id = self._node_to_id[output_node]
                    in_degree[output_node_id] -= 1

            depth_count += 1
        self._total_topo_depth = depth_count

        # Find the model input nodes
        for node in self._nodes:
            if self._is_model_inputs(node):
                self._model_inputs.append(node)

        self._model_inputs = sorted(self._model_inputs,
                                    key=lambda x: self._node_to_id[x])

    def _search_network(self, input_node, outputs, in_stack_nodes,
                        visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for hypermodel in input_node.out_hypermodels:
            for output_node in hypermodel.outputs:
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

    def _add_hypermodel(self, hypermodel):
        if hypermodel not in self._hypermodels:
            hypermodel_id = len(self._hypermodels)
            self._hypermodel_to_id[hypermodel] = hypermodel_id
            self._hypermodels.append(hypermodel)

    def _add_node(self, input_node):
        if input_node not in self._node_to_id:
            self._node_to_id[input_node] = len(self._node_to_id)

    def _get_metrics(self):
        metrics = []
        for metrics_list in [output_node.in_hypermodels[0].metrics for
                             output_node in self.outputs
                             if isinstance(output_node.in_hypermodels[0],
                                           head.HyperHead)]:
            metrics += metrics_list
        return metrics

    def _get_loss(self):
        loss = nest.flatten([output_node.in_hypermodels[0].loss
                             for output_node in self.outputs
                             if isinstance(output_node.in_hypermodels[0],
                                           head.HyperHead)])
        return loss

    def _compiled(self, hp, model):
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
        if self._plain_graph_hm:
            return self._plain_graph_hm.preprocess(hp, dataset, validation_data, fit)
        for hypermodel in self._hypermodels:
            if isinstance(hypermodel, processor.HyperPreprocessor):
                hypermodel.set_hp(hp)
        dataset = self._preprocess(dataset, fit=fit)
        if not validation_data:
            return dataset
        validation_data = self._preprocess(validation_data)
        return dataset, validation_data

    def _preprocess(self, dataset, fit=False):
        # Put hypermodels in groups by topological depth.
        # TODO: Dynamically process the topology instead of pregenerated. The
        # topology may change due to different hp. Some block may require multiple
        # rounds, some may require zero round.
        hypermodels_by_depth = []
        for depth in range(self._total_topo_depth):
            temp_hypermodels = []
            for hypermodel in self._hypermodels:
                if (self._hypermodel_topo_depth[
                    self._hypermodel_to_id[hypermodel]] == depth and
                        isinstance(hypermodel, processor.HyperPreprocessor)):
                    temp_hypermodels.append(hypermodel)
            if not temp_hypermodels:
                break
            hypermodels_by_depth.append(temp_hypermodels)

        # A list of input node ids in the same order as the x in the dataset.
        input_node_ids = [self._node_to_id[input_node] for input_node in self.inputs]

        # Iterate the depth.
        for hypermodels in hypermodels_by_depth:
            if fit:
                # Iterate the dataset to fit the preprocessors in current depth.
                for x, _ in dataset:
                    x = nest.flatten(x)
                    node_id_to_data = {
                        node_id: temp_x
                        for temp_x, node_id in zip(x, input_node_ids)
                    }
                    for hypermodel in hypermodels:
                        data = [node_id_to_data[self._node_to_id[input_node]]
                                for input_node in hypermodel.inputs]
                        hypermodel.update(data)

            for hypermodel in hypermodels:
                hypermodel.finalize()
                nest.flatten(hypermodel.outputs)[0].shape = hypermodel.output_shape

            # Transform the dataset.
            dataset = dataset.map(functools.partial(
                self._preprocess_transform,
                input_node_ids=input_node_ids,
                hypermodels=hypermodels))

            # Build input_node_ids for next depth.
            input_node_ids = list(sorted([self._node_to_id[hypermodel.outputs[0]]
                                          for hypermodel in hypermodels]))
        return dataset

    def _preprocess_transform(self, x, y, input_node_ids, hypermodels):
        x = nest.flatten(x)
        id_to_data = {
            node_id: temp_x
            for temp_x, node_id in zip(x, input_node_ids)
        }
        output_data = {}
        # Keep the Keras Model inputs even they are not inputs to the hypermodels.
        for node_id, data in id_to_data.items():
            if self._is_model_inputs(self._nodes[node_id]):
                output_data[node_id] = data
        # Transform each x by the corresponding hypermodel
        for hm in hypermodels:
            data = [id_to_data[self._node_to_id[input_node]]
                    for input_node in hm.inputs]
            data = tf.py_function(hm.transform,
                                  inp=nest.flatten(data),
                                  Tout=hm.output_types())
            data = nest.flatten(data)[0]
            data.set_shape(hm.output_shape)
            output_data[self._node_to_id[hm.outputs[0]]] = data
        return tuple(map(
            lambda index: output_data[index], sorted(output_data))), y

    @staticmethod
    def _is_model_inputs(node):
        for hypermodel in node.in_hypermodels:
            if not isinstance(hypermodel, processor.HyperPreprocessor):
                return False
        for hypermodel in node.out_hypermodels:
            if not isinstance(hypermodel, processor.HyperPreprocessor):
                return True
        return False
