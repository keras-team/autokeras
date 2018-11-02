from autokeras.constant import Constant
from autokeras.nn.graph import Graph
from autokeras.nn.layers import StubBatchNormalization, StubConv, StubDropout, StubPooling, StubDense, StubFlatten, \
    StubReLU
from abc import abstractmethod


class NetworkGenerator:
    def __init__(self, n_output_node, input_shape):
        self.n_output_node = n_output_node
        self.input_shape = input_shape

    @abstractmethod
    def generate(self, model_len, model_width):
        pass


class CnnGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(CnnGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')

    def generate(self, model_len=Constant.MODEL_LEN, model_width=Constant.MODEL_WIDTH):
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(StubConv(temp_input_channel, model_width, kernel_size=3), output_node_id)
            output_node_id = graph.add_layer(StubBatchNormalization(model_width), output_node_id)
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(StubPooling(), output_node_id)

        output_node_id = graph.add_layer(StubFlatten(), output_node_id)
        output_node_id = graph.add_layer(StubDropout(Constant.CONV_DROPOUT_RATE), output_node_id)
        output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], model_width),
                                         output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(model_width, self.n_output_node), output_node_id)
        return graph


class MlpGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 1:
            raise ValueError('The input dimension is too high.')

    def generate(self, model_len=Constant.MLP_MODEL_LEN, model_width=Constant.MLP_MODEL_WIDTH):
        if type(model_width) is list and not len(model_width) == model_len:
            raise ValueError('The length of \'model_width\' does not match \'model_len\'')
        elif type(model_width) is int:
            model_width = [model_width] * model_len

        graph = Graph(self.input_shape[0], False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for width in model_width:
            output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, width), output_node_id)
            output_node_id = graph.add_layer(StubDropout(Constant.MLP_DROPOUT_RATE), output_node_id)
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = width

        graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
        return graph
