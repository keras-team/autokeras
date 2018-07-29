from autokeras.generator import DefaultClassifierGenerator
from autokeras.layer_transformer import *
from tests.common import get_conv_dense_model


def test_deeper_conv_block():
    graph = DefaultClassifierGenerator(10, (28, 28, 3)).generate()
    layers = deeper_conv_block(graph.layer_list[1], 3)
    assert len(layers) == constant.CONV_BLOCK_DISTANCE + 1


def test_dense_to_deeper_layer():
    a = StubDense(100, 100)
    assert len(dense_to_deeper_block(a)) == 3


def test_dense_to_wider_layer():
    a = StubDense(10, 5)
    a.set_weights((np.random.rand(10, 5), np.random.rand(5)))
    b = StubDense(5, 10)
    b.set_weights((np.random.rand(5, 10), np.random.rand(10)))

    assert isinstance(wider_pre_dense(a, 5), StubDense)
    assert isinstance(wider_next_dense(b, 10, 10, 5), StubDense)


def test_wider_bn():
    bn_layer = StubBatchNormalization(3)
    bn_layer.set_weights([np.ones(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.ones(3, dtype=np.float32)])
    new_bn_layer = wider_bn(bn_layer, 1, 3, 4)
    assert new_bn_layer.get_weights()[0].shape[0] == 7


def test_wider_next_dense():
    real_layer = get_conv_dense_model().layers[6]
    layer = StubDense(real_layer.units, 'relu')
    layer.set_weights(real_layer.get_weights())
    new_layer = wider_next_dense(layer, 3, 3, 3)
    assert new_layer.get_weights()[0].shape == (6, 5)


def test_wider_conv():
    model = DefaultClassifierGenerator(10, (28, 28, 3)).generate().produce_model()
    model.set_weight_to_graph()
    graph = model.graph

    assert isinstance(wider_pre_conv(graph.layer_list[1], 3), StubConv)
    assert isinstance(wider_bn(graph.layer_list[2], 3, 3, 3), StubBatchNormalization)
    assert isinstance(wider_next_conv(graph.layer_list[6], 3, 3, 3), StubConv)
