from abc import abstractmethod

from autokeras.constant import Constant
from autokeras.nn.graph import Graph
from autokeras.nn.layers import StubDense, StubReLU, get_conv_class, get_dropout_class, \
    get_global_avg_pooling_class, get_pooling_class, get_batch_norm_class, StubDropout1d


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
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=Constant.MODEL_LEN, model_width=Constant.MODEL_WIDTH):
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=3), output_node_id)
            output_node_id = graph.add_layer(self.batch_norm(model_width), output_node_id)
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(Constant.CONV_DROPOUT_RATE), output_node_id)
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

        graph = Graph(self.input_shape, False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for width in model_width:
            output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, width), output_node_id)
            output_node_id = graph.add_layer(StubDropout1d(Constant.MLP_DROPOUT_RATE), output_node_id)
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = width

        graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
        return graph
