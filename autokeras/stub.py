from keras.engine import InputLayer
from keras.layers import Dense, Concatenate, BatchNormalization, Activation, Flatten, Dropout

from autokeras.layers import WeightedAdd, StubLayer, StubBatchNormalization, StubDense, StubConv, StubConcatenate, \
    StubWeightedAdd, StubActivation, StubPooling, StubDropout, StubFlatten
from autokeras.utils import is_conv_layer, is_pooling_layer, get_int_tuple


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

        if is_conv_layer(layer):
            temp_stub_layer = StubConv(layer.filters, layer.kernel_size, layer.__class__, input_id, output_id)
        elif isinstance(layer, Dense):
            temp_stub_layer = StubDense(layer.units, layer.activation, input_id, output_id)
        elif isinstance(layer, WeightedAdd):
            temp_stub_layer = StubWeightedAdd(input_id, output_id)
        elif isinstance(layer, Concatenate):
            temp_stub_layer = StubConcatenate(input_id, output_id)
        elif isinstance(layer, BatchNormalization):
            temp_stub_layer = StubBatchNormalization(input_id, output_id)
        elif isinstance(layer, Activation):
            temp_stub_layer = StubActivation(layer.activation, input_id, output_id)
        elif isinstance(layer, InputLayer):
            temp_stub_layer = StubLayer(input_id, output_id)
        elif isinstance(layer, Flatten):
            temp_stub_layer = StubFlatten(input_id, output_id)
        elif isinstance(layer, Dropout):
            temp_stub_layer = StubDropout(layer.rate, input_id, output_id)
        elif is_pooling_layer(layer):
            temp_stub_layer = StubPooling(layer.__class__, input_id, output_id)
        else:
            raise TypeError("The layer {} is illegal.".format(layer))
        if weighted:
            temp_stub_layer.set_weights(layer.get_weights())
        ret.add_layer(temp_stub_layer)
    ret.inputs = [tensor_dict[model.inputs[0]]]
    ret.outputs = [tensor_dict[model.outputs[0]]]
    return ret


class StubTensor:
    def __init__(self, shape=None):
        self.shape = shape
