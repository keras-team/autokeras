import os
import numpy as np

from autokeras.constant import Constant
from autokeras.graph import Graph
from autokeras.layers import StubReLU, StubConv, StubBatchNormalization, StubDropout, StubFlatten, StubSoftmax, \
    StubDense, StubConcatenate, StubAdd, StubPooling
from autokeras.preprocessor import DataTransformer


def get_concat_skip_model():
    graph = Graph((32, 32, 3), False)
    output_node_id = 0

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubConcatenate(), [output_node_id, temp_node_id])
    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(6, 3, 1), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubConcatenate(), [output_node_id, temp_node_id])
    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(6, 3, 1), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout(Constant.CONV_DROPOUT_RATE), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], 5),
                                     output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(5, 5), output_node_id)
    graph.add_layer(StubSoftmax(), output_node_id)

    graph.produce_model().set_weight_to_graph()

    return graph


def get_add_skip_model():
    graph = Graph((32, 32, 3), False)
    output_node_id = 0

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    temp_node_id = graph.add_layer(StubReLU(), temp_node_id)
    temp_node_id = graph.add_layer(StubConv(3, 3, 1), temp_node_id)
    temp_node_id = graph.add_layer(StubBatchNormalization(3), temp_node_id)
    output_node_id = graph.add_layer(StubAdd(), [output_node_id, temp_node_id])

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    temp_node_id = graph.add_layer(StubReLU(), temp_node_id)
    temp_node_id = graph.add_layer(StubConv(3, 3, 1), temp_node_id)
    temp_node_id = graph.add_layer(StubBatchNormalization(3), temp_node_id)
    output_node_id = graph.add_layer(StubAdd(), [output_node_id, temp_node_id])

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout(Constant.CONV_DROPOUT_RATE), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], 5),
                                     output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(5, 5), output_node_id)
    graph.add_layer(StubSoftmax(), output_node_id)

    graph.produce_model().set_weight_to_graph()

    return graph


def get_conv_data():
    return np.random.rand(1, 3, 32, 32)


def get_conv_dense_model():
    graph = Graph((32, 32, 3), False)
    output_node_id = 0

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout(Constant.DENSE_DROPOUT_RATE), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], 5),
                                     output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(5, 5), output_node_id)
    graph.add_layer(StubSoftmax(), output_node_id)

    graph.produce_model().set_weight_to_graph()

    return graph


def get_pooling_model():
    graph = Graph((32, 32, 3), False)
    output_node_id = 0

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubPooling(2), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout(Constant.CONV_DROPOUT_RATE), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], 5),
                                     output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(5, 5), output_node_id)
    graph.add_layer(StubSoftmax(), output_node_id)

    graph.produce_model().set_weight_to_graph()

    return graph


def get_regression_dataloaders():
    x_train = np.random.rand(200, 28, 28, 3)
    y_train = np.random.rand(200, 1)
    x_test = np.random.rand(190, 28, 28, 3)
    y_test = np.random.rand(190, 1)
    data_transformer = DataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    return train_data, test_data


def get_classification_dataloaders():
    x_train = np.random.rand(200, 28, 28, 3)
    y_train = np.random.rand(200, 3)
    x_test = np.random.rand(190, 28, 28, 3)
    y_test = np.random.rand(190, 3)
    data_transformer = DataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    return train_data, test_data


def clean_dir(path):
    for f in os.listdir(path):
        if f != '.gitkeep':
            os.remove(os.path.join(path, f))


class MockProcess(object):
    def __init__(self, target=None, args=None):
        self.target = target
        self.args = args
        self.result = None

    def join(self):
        pass

    def start(self):
        self.target(*self.args)

    def map_async(self, a, b):
        self.result = a(b[0])
        return self

    def get(self, timeout=None):
        return [self.result]

    def terminate(self):
        pass
