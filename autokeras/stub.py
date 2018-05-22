from autokeras.layers import to_stub_layer
from autokeras.utils import get_int_tuple


class StubModel:
    def __init__(self):
        self.layers = []
        self.input_shape = None
        self.inputs = []
        self.outputs = []

    def add_layer(self, layer):
        self.layers.append(layer)


def to_stub_model(model, weighted=False):
    node_count = 0
    tensor_dict = {}
    ret = StubModel()
    ret.input_shape = model.input_shape
    for layer in model.layers:
        if isinstance(layer.input, list):
            input_nodes = layer.input
        else:
            input_nodes = [layer.input]

        for node in input_nodes + [layer.output]:
            if node not in tensor_dict:
                tensor_dict[node] = StubTensor(get_int_tuple(node.shape))
                node_count += 1

        if isinstance(layer.input, list):
            input_id = []
            for node in layer.input:
                input_id.append(tensor_dict[node])
        else:
            input_id = tensor_dict[layer.input]
        output_id = tensor_dict[layer.output]

        temp_stub_layer = to_stub_layer(layer, input_id, output_id)
        if weighted:
            temp_stub_layer.set_weights(layer.get_weights())
        ret.add_layer(temp_stub_layer)
    ret.inputs = [tensor_dict[model.inputs[0]]]
    ret.outputs = [tensor_dict[model.outputs[0]]]
    return ret


class StubTensor:
    def __init__(self, shape=None):
        self.shape = shape
