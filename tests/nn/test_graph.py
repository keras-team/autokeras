from autokeras.nn.generator import CnnGenerator
from autokeras.nn.graph import *
from autokeras.net_transformer import legal_graph
from tests.common import get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model


def test_conv_deeper_stub():
    graph = get_conv_dense_model()
    layer_num = graph.n_layers
    graph.to_conv_deeper_model(4, 3)

    assert graph.n_layers == layer_num + 3


def test_conv_deeper():
    graph = get_conv_dense_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_conv_deeper_model(4, 3)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()
    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-1


def test_dense_deeper_stub():
    graph = get_conv_dense_model()
    graph.weighted = False
    layer_num = graph.n_layers
    graph.to_dense_deeper_model(9)

    assert graph.n_layers == layer_num + 2


def test_dense_deeper():
    graph = get_conv_dense_model()
    model = graph.produce_model()
    graph = deepcopy(graph)
    graph.to_dense_deeper_model(9)
    new_model = graph.produce_model()
    input_data = torch.Tensor(get_conv_data())

    model.eval()
    new_model.eval()
    output1 = model(input_data)
    output2 = new_model(input_data)

    assert (output1 - output2).abs().sum() < 1e-3


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

    assert graph.n_layers == layer_num + 5


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

    assert graph.n_layers == layer_num + 5


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
    assert descriptor.n_conv == 5
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.ADD_CONNECT), (3, 4, NetworkDescriptor.ADD_CONNECT)]


def test_extract_descriptor_concat():
    descriptor = get_concat_skip_model().extract_descriptor()
    assert descriptor.n_conv == 5
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.CONCAT_CONNECT),
                                           (3, 4, NetworkDescriptor.CONCAT_CONNECT)]


def test_deep_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.deep_layer_ids()) == 3


def test_wide_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.wide_layer_ids()) == 2


def test_skip_connection_layer_ids():
    graph = get_conv_dense_model()
    assert len(graph.skip_connection_layer_ids()) == 1


def test_wider_dense():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    graph.produce_model().set_weight_to_graph()
    history = [('to_wider_model', 14, 64)]
    for args in history:
        getattr(graph, args[0])(*list(args[1:]))
        graph.produce_model()
    assert legal_graph(graph)


def test_long_transform():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    history = [('to_wider_model', 1, 256), ('to_conv_deeper_model', 1, 3),
               ('to_concat_skip_model', 5, 9)]
    for args in history:
        getattr(graph, args[0])(*list(args[1:]))
        graph.produce_model()
    assert legal_graph(graph)


def test_node_consistency():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    assert graph.layer_list[6].output.shape == (16, 16, 64)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape

    graph.to_wider_model(5, 64)
    assert graph.layer_list[5].output.shape == (16, 16, 128)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape

    graph.to_conv_deeper_model(5, 3)
    assert graph.layer_list[19].output.shape == (16, 16, 128)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape

    graph.to_add_skip_model(5, 18)
    assert graph.layer_list[23].output.shape == (16, 16, 128)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape

    graph.to_concat_skip_model(5, 18)
    assert graph.layer_list[25].output.shape == (16, 16, 256)

    for layer in graph.layer_list:
        assert layer.output.shape == layer.output_shape


def test_produce_keras_model():
    for graph in [get_conv_dense_model(),
                  get_add_skip_model(),
                  get_pooling_model(),
                  get_concat_skip_model()]:
        model = graph.produce_keras_model()
        assert isinstance(model, keras.models.Model)


def test_keras_model():
    for graph in [get_conv_dense_model(),
                  get_add_skip_model(),
                  get_pooling_model(),
                  get_concat_skip_model()]:
        keras_model = KerasModel(graph)
        keras_model.set_weight_to_graph()
        assert isinstance(keras_model, KerasModel)


def test_graph_size():
    graph = CnnGenerator(10, (32, 32, 3)).generate()
    assert graph.size() == 7498
