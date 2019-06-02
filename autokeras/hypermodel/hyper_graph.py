from queue import Queue
import tensorflow as tf

from autokeras import HyperModel
from autokeras.hypermodel.hyper_head import ClassificationHead
from autokeras.layer_utils import format_inputs


class HyperGraph(HyperModel):
    def __init__(self,
                 inputs,
                 outputs,
                 **kwargs):
        super().__init__(**kwargs)
        self.inputs = format_inputs(inputs, self.name)
        self.outputs = format_inputs(outputs, self.name)
        self.node_to_id = None
        self.nodes = None
        self.hypermodel_to_id = None
        self.hypermodels = None
        self._build_network()

    def build(self, hp):
        real_nodes = {}
        for input_node in self.inputs:
            node_id = self.node_to_id[input_node]
            real_nodes[node_id] = input_node.build(hp)
        for hypermodel in self.hypermodels:
            outputs = hypermodel.build(hp,
                                       inputs=[real_nodes[self.node_to_id[input_node]]
                                               for input_node in hypermodel.inputs],
                                       sub_model=True)
            outputs = format_inputs(outputs, hypermodel.name)
            for output_node, real_output_node in zip(hypermodel.outputs, outputs):
                real_nodes[self.node_to_id[output_node]] = real_output_node
        model = tf.keras.Model([real_nodes[self.node_to_id[input_node]] for input_node in self.inputs],
                               [real_nodes[self.node_to_id[output_node]] for output_node in self.outputs])
        # Specify hyperparameters from compile(...)
        optimizer = hp.Choice('optimizer',
                                   [tf.keras.optimizers.Adam,
                                    tf.keras.optimizers.Adadelta,
                                    tf.keras.optimizers.SGD])()
        metrics = self._infer_metrics()
        loss = self._infer_loss()

        model.compile(optimizer=optimizer,
                      metrics=metrics,
                      loss=loss)

        return model

    def _build_network(self):
        self.node_to_id = {}

        # Recursively find all the interested nodes.
        for input_node in self.inputs:
            self._search_network(input_node, self.outputs, set(), set())
        self.nodes = sorted(list(self.node_to_id.keys()), key=lambda x: self.node_to_id[x])

        for node in (self.inputs + self.outputs):
            if node not in self.node_to_id:
                raise ValueError("Inputs and outputs not connected.")

        # Find the hypermodels and sort the hypermodels in topological order.
        self.hypermodels = []
        self.hypermodel_to_id = {}
        visited_nodes = set()
        queue = Queue()
        for input_node in self.inputs:
            queue.put(input_node)
            visited_nodes.add(input_node)
        while not queue.empty():
            input_node = queue.get()
            for hypermodel in input_node.out_hypermodels:
                # Check at least one output node of the hypermodel is in the interested nodes.
                if not any([output_node in self.node_to_id for output_node in hypermodel.outputs]):
                    continue
                self._add_hypermodel(hypermodel)
                for output_node in hypermodel.outputs:
                    # The node is not visited and in interested nodes.
                    if output_node not in visited_nodes and output_node in self.node_to_id:
                        visited_nodes.add(output_node)
                        queue.put(output_node)
        for output_node in self.outputs:
            hypermodel = output_node.in_hypermodels[0]
            hypermodel.output_shape = output_node.shape

    def _search_network(self, input_node, outputs, in_stack_nodes, visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for hypermodel in input_node.out_hypermodels:
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

    def _infer_metrics(self):
        if any([isinstance(hypermodel, ClassificationHead) for hypermodel in self.hypermodels]):
            return [tf.keras.metrics.Accuracy]
        return [tf.keras.metrics.mse]

    def _infer_loss(self):
        if any([isinstance(hypermodel, ClassificationHead) for hypermodel in self.hypermodels]):
            return tf.keras.losses.categorical_crossentropy
        return tf.keras.losses.mean_squared_error
