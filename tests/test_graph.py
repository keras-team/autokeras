import numpy as np
from autokeras.graph import *
from tests.common import get_conv_model, get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model


def test_graph():
    graph = Graph(get_conv_model())
    assert graph.n_nodes == 9


def test_conv_deeper_stub():
    model = get_conv_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_conv_deeper_model(4, 4)

    assert graph.n_layers == layer_num + 4


def test_conv_deeper():
    model = get_conv_model()
    graph = Graph(model, True)
    graph.to_conv_deeper_model(4, 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_dense_deeper_stub():
    model = get_conv_dense_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_dense_deeper_model(5)

    assert graph.n_layers == layer_num + 2


def test_dense_deeper():
    model = get_conv_dense_model()
    graph = Graph(model, True)
    graph.to_dense_deeper_model(5)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_conv_wider_stub():
    model = get_add_skip_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_wider_model(8, 3)

    assert graph.n_layers == layer_num


def test_conv_wider():
    model = get_concat_skip_model()
    graph = Graph(model, True)
    graph.to_wider_model(4, 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 4e-1


def test_dense_wider_stub():
    model = get_add_skip_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_wider_model(19, 3)

    assert graph.n_layers == layer_num


def test_dense_wider():
    model = get_add_skip_model()
    graph = Graph(model, True)
    graph.to_wider_model(19, 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_skip_add_over_pooling_stub():
    model = get_pooling_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_add_skip_model(0, 10)

    assert graph.n_layers == layer_num + 2


def test_skip_add_over_pooling():
    model = get_pooling_model()
    graph = Graph(model, True)
    graph.to_add_skip_model(0, 10)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_concat_over_pooling_stub():
    model = get_pooling_model()
    graph = Graph(model, False)
    layer_num = graph.n_layers
    graph.to_concat_skip_model(0, 13)

    assert graph.n_layers == layer_num + 2


def test_skip_concat_over_pooling():
    model = get_pooling_model()
    graph = Graph(model, True)
    graph.to_concat_skip_model(4, 9)
    graph.to_concat_skip_model(4, 9)
    graph = Graph(graph.produce_model(), True)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 4e-1


def test_extract_descriptor_add():
    model = get_add_skip_model()
    descriptor = Graph(model).extract_descriptor()
    assert descriptor.n_conv == 4
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.ADD_CONNECT), (3, 4, NetworkDescriptor.ADD_CONNECT)]


def test_extract_descriptor_concat():
    model = get_concat_skip_model()
    descriptor = Graph(model).extract_descriptor()
    assert descriptor.n_conv == 5
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.CONCAT_CONNECT),
                                           (3, 4, NetworkDescriptor.CONCAT_CONNECT)]


def test_deep_layer_ids():
    model = get_conv_dense_model()
    graph = Graph(model, True)
    assert len(graph.deep_layer_ids()) == 2


def test_wide_layer_ids():
    model = get_conv_dense_model()
    graph = Graph(model, True)
    assert len(graph.wide_layer_ids()) == 1


def test_skip_connection_layer_ids():
    model = get_conv_dense_model()
    graph = Graph(model, True)
    assert len(graph.skip_connection_layer_ids()) == 0
