from autokeras.layer_transformer import *
from autokeras.stub import to_stub_model
from tests.common import get_conv_model, get_add_skip_model, get_conv_dense_model


def test_deeper_conv_block():
    model = to_stub_model(get_conv_model(), True)
    layers = deeper_conv_block(model.layers[1], 3)
    assert len(layers) == constant.CONV_BLOCK_SIZE


def test_dense_to_deeper_layer():
    a = StubDense(100, 'relu')
    assert len(dense_to_deeper_block(a)) == 2


def test_dense_to_wider_layer():
    a = StubDense(5, 'relu')
    a.set_weights((np.random.rand(10, 5), np.random.rand(5)))
    b = StubDense(10, 'relu')
    b.set_weights((np.random.rand(5, 10), np.random.rand(10)))

    assert isinstance(wider_pre_dense(a, 5), StubDense)
    assert isinstance(wider_next_dense(b, 10, 10, 5), StubDense)


def test_wider_bn():
    bn_layer = StubBatchNormalization()
    bn_layer.set_weights(get_conv_model().layers[2].get_weights())
    print(bn_layer.get_weights())
    new_bn_layer = wider_bn(bn_layer, 1, 3, 4)
    assert new_bn_layer.get_weights()[0].shape[0] == 7


def test_wider_weighted_add():
    layer = StubWeightedAdd()
    layer.set_weights(get_add_skip_model().layers[13].get_weights())
    new_layer = wider_weighted_add(layer, 4)
    assert isinstance(new_layer, StubWeightedAdd)


def test_wider_next_dense():
    real_layer = get_conv_dense_model().layers[6]
    layer = StubDense(real_layer.units, 'relu')
    layer.set_weights(real_layer.get_weights())
    new_layer = wider_next_dense(layer, 3, 3, 3)
    assert new_layer.get_weights()[0].shape == (150, 5)


def test_wider_conv():
    model = to_stub_model(get_conv_model(), True)

    assert isinstance(wider_pre_conv(model.layers[1], 3), StubConv)
    assert isinstance(wider_bn(model.layers[2], 3, 3, 3), StubBatchNormalization)
    assert isinstance(wider_next_conv(model.layers[5], 3, 3, 3), StubConv)
