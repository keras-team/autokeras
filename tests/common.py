from time import sleep

import os
import queue
from copy import deepcopy

import numpy as np

from autokeras.constant import Constant
from autokeras.nn.graph import Graph
from autokeras.nn.layers import StubReLU, StubFlatten, StubSoftmax, StubDense, StubConcatenate, StubAdd, \
    StubConv2d, StubBatchNormalization2d, StubDropout2d
from autokeras.nn.layers import StubPooling2d
from autokeras.preprocessor import ImageDataTransformer, DataTransformerMlp

TEST_TEMP_AUTO_KERAS_DIR = 'tests/resources/temp/autokeras'
TEST_TEMP_DIR = 'tests/resources/temp'


def get_concat_skip_model():
    graph = Graph((32, 32, 3), False)
    output_node_id = 0

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubConcatenate(), [output_node_id, temp_node_id])
    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(6, 3, 1), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubConcatenate(), [output_node_id, temp_node_id])
    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(6, 3, 1), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout2d(Constant.CONV_DROPOUT_RATE), output_node_id)

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
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    temp_node_id = graph.add_layer(StubReLU(), temp_node_id)
    temp_node_id = graph.add_layer(StubConv2d(3, 3, 1), temp_node_id)
    temp_node_id = graph.add_layer(StubBatchNormalization2d(3), temp_node_id)
    output_node_id = graph.add_layer(StubAdd(), [output_node_id, temp_node_id])

    temp_node_id = output_node_id

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    temp_node_id = graph.add_layer(StubReLU(), temp_node_id)
    temp_node_id = graph.add_layer(StubConv2d(3, 3, 1), temp_node_id)
    temp_node_id = graph.add_layer(StubBatchNormalization2d(3), temp_node_id)
    output_node_id = graph.add_layer(StubAdd(), [output_node_id, temp_node_id])

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout2d(Constant.CONV_DROPOUT_RATE), output_node_id)

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
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout2d(Constant.DENSE_DROPOUT_RATE), output_node_id)

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
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubPooling2d(2), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubConv2d(3, 3, 3), output_node_id)
    output_node_id = graph.add_layer(StubBatchNormalization2d(3), output_node_id)

    output_node_id = graph.add_layer(StubFlatten(), output_node_id)
    output_node_id = graph.add_layer(StubDropout2d(Constant.CONV_DROPOUT_RATE), output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], 5),
                                     output_node_id)

    output_node_id = graph.add_layer(StubReLU(), output_node_id)
    output_node_id = graph.add_layer(StubDense(5, 5), output_node_id)
    graph.add_layer(StubSoftmax(), output_node_id)

    graph.produce_model().set_weight_to_graph()

    return graph


def get_regression_data_loaders():
    x_train = np.random.rand(200, 28, 28, 3)
    y_train = np.random.rand(200, 1)
    x_test = np.random.rand(190, 28, 28, 3)
    y_test = np.random.rand(190, 1)
    data_transformer = ImageDataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    return train_data, test_data


def get_classification_data_loaders():
    x_train = np.random.rand(200, 28, 28, 3)
    y_train = np.random.rand(200, 3)
    x_test = np.random.rand(190, 28, 28, 3)
    y_test = np.random.rand(190, 3)
    data_transformer = ImageDataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    return train_data, test_data


def get_classification_train_data_loaders():
    x_train = np.random.rand(200, 32, 32, 3)
    data_transformer = ImageDataTransformer(x_train, augment=True)
    train_data = data_transformer.transform_train(x_train)
    return train_data


def get_classification_data_loaders_mlp():
    x_train = np.random.rand(200, 28)
    y_train = np.random.rand(200, 3)
    x_test = np.random.rand(190, 28)
    y_test = np.random.rand(190, 3)
    data_transformer = DataTransformerMlp(x_train)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    return train_data, test_data


def clean_dir(path):
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if f != '.gitkeep':
            if os.path.isfile(full_path):
                os.remove(full_path)
            else:
                os.rmdir(full_path)


class MockProcess(object):
    def __init__(self, target=None, args=None):
        self.target = target
        self.args = args
        self.result = None
        self.count = 0

    def join(self):
        pass

    def start(self):
        self.result = self.target(*self.args)

    def map_async(self, a, b):
        self.result = a(b[0])
        return self

    def get(self, timeout=None):
        str(timeout)
        return [self.result]

    def get_context(self, start_method='fork'):
        return self

    def Queue(self):
        class MockQueue(queue.Queue):
            def __init__(self):
                super().__init__()
                self.count = 0

            def qsize(self):
                self.count += 1
                if self.count > 8:
                    return 1
                return 0

        # (0.5, 0.8, get_pooling_model())
        return MockQueue()

    def Process(self, target, args):
        self.target = target
        self.args = args
        return self

    def terminate(self):
        pass

    def close(self):
        pass


def simple_transform(graph):
    graph.to_wider_model(6, 64)
    return [deepcopy(graph)]


def simple_transform_mlp(graph):
    graph.to_wider_model(3, 64)
    return [deepcopy(graph)]


def mock_train(**kwargs):
    str(kwargs)
    sleep(0.1)
    return 1, 0


def mock_exception_handling_train(**kwargs):
    str(kwargs)
    raise Exception


def mock_out_of_memory_train(**kwargs):
    str(kwargs)
    raise RuntimeError('CUDA: out of memory.')

# def mock_nvidia_smi_output(*arg, **kwargs):
#     return \
#         '    Free                        : 1 MiB \n' \
#         '    Free                        : 11176 MiB \n' \
#         '    Free                        : 1 MiB \n' \
#         '    Free                        : 1 MiB'
