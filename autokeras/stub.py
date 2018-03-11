from keras.engine import InputLayer
from keras.layers import Dense, Concatenate, BatchNormalization, Activation, Flatten

from autokeras.layers import WeightedAdd
from autokeras.utils import is_conv_layer


class StubLayer:
    def __init__(self, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node


class StubBatchNormalization(StubLayer):
    pass


class StubDense(StubLayer):
    def __init__(self, units, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.units = units


class StubConv(StubLayer):
    def __init__(self, filters, input_node=None, output_node=None):
        super().__init__(input_node, output_node)
        self.filters = filters


class StubAggregateLayer(StubLayer):
    def __init__(self, input_nodes=None, output_node=None):
        if input_nodes is None:
            input_nodes = []
        super().__init__(input_nodes, output_node)


class StubConcatenate(StubAggregateLayer):
    pass


class StubWeightedAdd(StubAggregateLayer):
    pass


class StubActivation(StubLayer):
    pass


class StubModel:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)


def to_stub_model(model):
    node_count = 0
    node_to_id = {}
    ret = StubModel()
    for layer in model.layers:
        if isinstance(layer.input, list):
            input_nodes = layer.input
        else:
            input_nodes = [layer.input]

        for node in input_nodes + [layer.output]:
            if node not in node_to_id:
                node_to_id[node] = node_count
                node_count += 1

        if isinstance(layer.input, list):
            input_id = []
            for node in layer.input:
                input_id.append(node_to_id[node])
        else:
            input_id = node_to_id[layer.input]
        output_id = node_to_id[layer.output]

        if is_conv_layer(layer):
            temp_stub_layer = StubConv(layer.filters, input_id, output_id)
        elif isinstance(layer, Dense):
            temp_stub_layer = StubDense(layer.units, input_id, output_id)
        elif isinstance(layer, WeightedAdd):
            temp_stub_layer = StubWeightedAdd(input_id, output_id)
        elif isinstance(layer, Concatenate):
            temp_stub_layer = StubConcatenate(input_id, output_id)
        elif isinstance(layer, BatchNormalization):
            temp_stub_layer = StubBatchNormalization(input_id, output_id)
        elif isinstance(layer, Activation):
            temp_stub_layer = StubActivation(input_id, output_id)
        elif isinstance(layer, InputLayer):
            temp_stub_layer = StubLayer(input_id, output_id)
        elif isinstance(layer, Flatten):
            temp_stub_layer = StubLayer(input_id, output_id)
        else:
            raise TypeError("The layer {} is illegal.".format(layer))
        ret.add_layer(temp_stub_layer)

    return ret
