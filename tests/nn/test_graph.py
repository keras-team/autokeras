import torch
from autokeras.nn.generator import CnnGenerator, ResNetGenerator, DenseNetGenerator
from autokeras.nn.graph import *
from tests.common import get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model


def test_conv_wider_stub():
    graph = get_add_skip_model()
    graph.weighted = False
    layer_num = graph.n_layers
    graph.to_wider_model(7, 3)

    assert graph.n_layers == layer_num


def test_conv_wider():
    graph = get_concat_skip_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_wider_model(4, 3)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()

    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-1


def test_dense_wider_stub():
    graph = get_add_skip_model()
    graph.weighted = False
    layer_num = graph.n_layers
    graph.to_wider_model(26, 3)

    assert graph.n_layers == layer_num


def test_dense_wider():
    graph = get_add_skip_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_wider_model(26, 3)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()

    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-4


def test_skip_add_over_pooling_stub():
    graph = get_pooling_model()
    graph.weighted = False
    layer_num = graph.n_layers
    graph.to_add_skip_model(1, 8)

    assert graph.n_layers == layer_num + 4


def test_skip_add_over_pooling():
    graph = get_pooling_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_add_skip_model(1, 8)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()

    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-4


def test_skip_concat_over_pooling_stub():
    graph = get_pooling_model()
    graph.weighted = False
    layer_num = graph.n_layers
    graph.to_concat_skip_model(1, 11)

    assert graph.n_layers == layer_num + 4


def test_skip_concat_over_pooling():
    graph = get_pooling_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_concat_skip_model(4, 8)
    graph.to_concat_skip_model(4, 8)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()

    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-4


def test_extract_descriptor_add():
    descriptor = get_add_skip_model().extract_descriptor()
    assert len(descriptor.layers) == 24
    assert descriptor.skip_connections == [(6, 10, NetworkDescriptor.ADD_CONNECT),
                                           (10, 14, NetworkDescriptor.ADD_CONNECT)]


def test_extract_descriptor_concat():
    descriptor = get_concat_skip_model().extract_descriptor()
    assert len(descriptor.layers) == 32
    assert descriptor.skip_connections == [(6, 10, NetworkDescriptor.CONCAT_CONNECT),
                                           (13, 17, NetworkDescriptor.CONCAT_CONNECT)]


def test_deep_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.deep_layer_ids()) == 13


def test_wide_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.wide_layer_ids()) == 2


def test_skip_connection_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.skip_connection_layer_ids()) == 12


def test_wider_dense():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    graph.produce_model().set_weight_to_graph()
    history = [('to_wider_model', 14, 64)]
    for args in history:
        getattr(graph, args[0])(*list(args[1:]))
        graph.produce_model()
    assert graph.layer_list[14].output.shape[-1] == 128


def test_node_consistency():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    assert graph.layer_list[6].output.shape == (16, 16, 64)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape

    graph.to_wider_model(6, 64)
    assert graph.layer_list[6].output.shape == (16, 16, 128)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape


def test_graph_size():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    assert graph.size() == 80982


def test_long_transform():
    graph = ResNetGenerator(10, (28, 28, 1)).generate()
    graph.to_deeper_model(16, StubReLU())
    graph.to_deeper_model(16, StubReLU())
    graph.to_add_skip_model(13, 47)
    model = graph.produce_model()
    model(torch.Tensor(np.random.random((10, 1, 28, 28))))


def test_long_transform2():
    graph = CnnGenerator(10, (28, 28, 1)).generate()
    graph.to_add_skip_model(2, 3)
    graph.to_concat_skip_model(2, 3)
    model = graph.produce_model()
    model(torch.Tensor(np.random.random((10, 1, 28, 28))))


# def test_long_transform3():
#     graph = DenseNetGenerator(10, (28, 28, 1)).generate()
#     for i in range(20):
#         graph = transform(graph)[3]
#     print(graph.operation_history)
#     model = graph.produce_model()
#     model(torch.Tensor(np.random.random((10, 1, 28, 28))))


def test_long_transform4():
    graph = ResNetGenerator(10, (28, 28, 1)).generate()
    graph.to_concat_skip_model(57, 68)
    model = graph.produce_model()
    model(torch.Tensor(np.random.random((10, 1, 28, 28))))


def test_long_transform5():
    graph = ResNetGenerator(10, (28, 28, 1)).generate()
    graph.to_concat_skip_model(19, 60)
    graph.to_wider_model(52, 256)
    model = graph.produce_model()
    model(torch.Tensor(np.random.random((10, 1, 28, 28))))


def test_long_transform6():
    graph = DenseNetGenerator(10, (28, 28, 1)).generate()
    graph.to_concat_skip_model(126, 457)
    model = graph.produce_model()
    model(torch.Tensor(np.random.random((10, 1, 28, 28))))
