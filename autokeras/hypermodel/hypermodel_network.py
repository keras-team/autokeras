from abc import ABC
from queue import Queue
import tensorflow as tf

from autokeras import HyperModel
from autokeras.hyperparameters import HyperParameters


class ConnectedHyperParameters(HyperParameters):
    def retrieve(self, name, type, config):
        super().retrieve(tf.get_default_graph().get_name_scope() + '/' + name, type, config)


class HyperNode(HyperModel):
    def __init__(self):
        super().__init__()
        self.in_hypermodel = []
        self.out_hypermodel = []

    def add_in_hypermodel(self, hypermodel):
        self.in_hypermodel.append(hypermodel)

    def add_out_hypermodel(self, hypermodel):
        self.out_hypermodel.append(hypermodel)

    def build(self, hp):
        raise NotImplementedError


class ConnectedHyperModel(HyperModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            prefix = self.__class__.__name__
            self.name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
        self.inputs = None
        self.outputs = None
        self.n_output_node = 1
        self.build = self._build_with_name_scope

    def __call__(self, inputs):
        self.inputs = inputs
        for input_node in inputs:
            input_node.add_out_hypermodel(self)
        self.outputs = []
        for _ in range(self.n_output_node):
            output_node = HyperNode()
            output_node.add_in_hypermodel(self)
            self.outputs.append(output_node)
        return self.outputs

    def build(self, hp, inputs=None):
        raise NotImplementedError

    def _build_with_name_scope(self, hp, inputs=None):
        with tf.name_scope(self.name):
            self.build(hp, inputs)


class HyperModelNetwork(ConnectedHyperModel):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self.node_to_id = None
        self.nodes = None
        self.hypermodel_to_id = None
        self.hypermodels = None
        self._build_network(inputs, outputs)

    def build(self, hp, inputs=None):
        real_nodes = {}
        for input_node in inputs:
            node_id = self.node_to_id[input_node]
            real_nodes[node_id] = input_node.build(hp)
        for hypermodel in self.hypermodels:
            model = hypermodel.build(hp, inputs=[self.node_to_id[input_node] for input_node in hypermodel.inputs])
            for output_node, real_output_node in zip(hypermodel.outputs, model.outputs):
                real_nodes[self.node_to_id[output_node]] = real_output_node
        return tf.keras.Model([real_nodes[self.node_to_id[input_node]] for input_node in self.inputs],
                              [real_nodes[self.node_to_id[output_node]] for output_node in self.outputs])

    def _build_network(self, inputs, outputs):
        self.node_to_id = {}

        # Recursively find all the interested nodes.
        for input_node in inputs:
            self._search_network(input_node, outputs, set(), set())
        self.nodes = sorted(list(self.node_to_id.keys()), key=lambda x: self.node_to_id[x])

        for node in (inputs + outputs):
            if node not in self.node_to_id:
                raise ValueError("Inputs and outputs not connected.")

        # Find the hypermodels and sort the hypermodels in topological order.
        self.hypermodels = []
        self.hypermodel_to_id = {}
        visited_nodes = set()
        queue = Queue()
        for input_node in inputs:
            queue.put(input_node)
            visited_nodes.add(input_node)
        while not queue.empty():
            input_node = queue.get()
            for hypermodel in input_node.out_hypermodel:
                # Check at least one output node of the hypermodel is in the interested nodes.
                if not any([output_node in self.node_to_id for output_node in hypermodel.outputs]):
                    continue
                self._add_hypermodel(hypermodel)
                for output_node in hypermodel.outputs:
                    # The node is not visited and in interested nodes.
                    if output_node not in visited_nodes and output_node in self.node_to_id:
                        visited_nodes.add(output_node)
                        queue.put(output_node)

    def _search_network(self, input_node, outputs, in_stack_nodes, visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for hypermodel in input_node.out_hypermodel:
            for output_node in hypermodel.outputs:
                if output_node in in_stack_nodes:
                    raise ValueError("The network has a cycle.")
                if output_node not in visited_nodes:
                    self._search_network(output_node, outputs, in_stack_nodes, visited_nodes)
                if output_node in self.node_to_id.keys():
                    outputs_reached = True

        if outputs_reached:
            self._add_node(input_node)

        in_stack_nodes.remove(input_node)

    def _add_hypermodel(self, hypermodel):
        if hypermodel not in self.hypermodels:
            hypermodel_id = len(self.hypermodels)
            self.hypermodel_to_id[hypermodel] = hypermodel_id
            self.hypermodels.append(hypermodel)
        for output_node in hypermodel.outputs:
            self._add_node(output_node)
        for input_node in hypermodel.inputs:
            if input_node not in self.node_to_id:
                raise ValueError("A required input is missing for HyperModel {name}. ".format(name=hypermodel.name))

    def _add_node(self, input_node):
        if input_node not in self.node_to_id:
            self.node_to_id[input_node] = len(self.node_to_id)
