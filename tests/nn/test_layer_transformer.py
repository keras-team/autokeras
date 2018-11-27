from autokeras.constant import Constant
from autokeras.nn.generator import CnnGenerator
from autokeras.nn.layer_transformer import *
from autokeras.nn.layers import StubBatchNormalization2d, StubConv2d
from tests.common import get_conv_dense_model


def test_dense_to_wider_layer():
    a = StubDense(10, 5)
    a.set_weights((np.random.rand(10, 5), np.random.rand(5)))
    b = StubDense(5, 10)
    b.set_weights((np.random.rand(5, 10), np.random.rand(10)))

    assert isinstance(wider_pre_dense(a, 5), StubDense)
    assert isinstance(wider_next_dense(b, 10, 10, 5), StubDense)


def test_wider_bn():
    bn_layer = StubBatchNormalization2d(3)
    bn_layer.set_weights([np.ones(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.ones(3, dtype=np.float32)])
    new_bn_layer = wider_bn(bn_layer, 1, 3, 4)
    assert new_bn_layer.get_weights()[0].shape[0] == 7


def test_wider_next_dense():
    real_layer = get_conv_dense_model().layer_list[9]
    layer = StubDense(real_layer.input_units, real_layer.units)
    layer.set_weights(real_layer.get_weights())
    new_layer = wider_next_dense(layer, 3, 3, 3)
    assert new_layer.get_weights()[0].shape == (5, 6144)


def test_wider_conv():
    model = CnnGenerator(10, (28, 28, 3)).generate().produce_model()
    model.set_weight_to_graph()
    graph = model.graph

    assert isinstance(wider_pre_conv(graph.layer_list[2], 3), StubConv2d)
    assert isinstance(wider_bn(graph.layer_list[5], 3, 3, 3), StubBatchNormalization2d)
    assert isinstance(wider_next_conv(graph.layer_list[6], 3, 3, 3), StubConv2d)
