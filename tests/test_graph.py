import numpy as np
from autokeras.graph import *
from autokeras.stub import to_stub_model
from tests.common import get_conv_model, get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model


def test_graph():
    graph = Graph(get_conv_model())
    assert graph.n_nodes == 9


def test_conv_deeper_stub():
    model = get_conv_model()
    graph = Graph(to_stub_model(model))
    layer_num = graph.n_layers
    graph.to_conv_deeper_model(4, 4)

    assert graph.n_layers == layer_num + 4


def test_conv_deeper():
    model = get_conv_model()
    graph = NetworkMorphismGraph(model)
    graph.to_conv_deeper_model(graph.layer_to_id[model.layers[5]], 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_dense_deeper_stub():
    model = to_stub_model(get_conv_dense_model())
    graph = Graph(model)
    layer_num = graph.n_layers
    graph.to_dense_deeper_model(graph.layer_to_id[model.layers[6]])

    assert graph.n_layers == layer_num + 2


def test_dense_deeper():
    model = get_conv_dense_model()
    graph = NetworkMorphismGraph(model)
    graph.to_dense_deeper_model(graph.layer_to_id[model.layers[6]])
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_conv_wider_stub():
    model = to_stub_model(get_add_skip_model())
    graph = Graph(model)
    layer_num = graph.n_layers
    graph.to_wider_model(graph.layer_to_id[model.layers[9]], 3)

    assert graph.n_layers == layer_num


def test_conv_wider():
    model = get_concat_skip_model()
    graph = NetworkMorphismGraph(model)
    graph.to_wider_model(graph.layer_to_id[model.layers[5]], 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 4e-1


def test_dense_wider_stub():
    model = to_stub_model(get_add_skip_model())
    graph = Graph(model)
    layer_num = graph.n_layers
    graph.to_wider_model(graph.layer_to_id[model.layers[-3]], 3)

    assert graph.n_layers == layer_num


def test_dense_wider():
    model = get_add_skip_model()
    graph = NetworkMorphismGraph(model)
    graph.to_wider_model(graph.layer_to_id[model.layers[-3]], 3)
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_skip_add_over_pooling_stub():
    model = to_stub_model(get_pooling_model())
    graph = Graph(model)
    layer_num = graph.n_layers
    graph.to_add_skip_model(graph.layer_to_id[model.layers[4]], graph.layer_to_id[model.layers[11]])

    assert graph.n_layers == layer_num + 2


def test_skip_add_over_pooling():
    model = get_pooling_model()
    graph = NetworkMorphismGraph(model)
    graph.to_add_skip_model(graph.layer_to_id[model.layers[4]], graph.layer_to_id[model.layers[11]])
    new_model = graph.produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_concat_over_pooling_stub():
    model = to_stub_model(get_pooling_model())
    graph = Graph(model)
    layer_num = graph.n_layers
    graph.to_concat_skip_model(graph.layer_to_id[model.layers[5]], graph.layer_to_id[model.layers[14]])

    assert graph.n_layers == layer_num + 2


def test_skip_concat_over_pooling():
    model = get_pooling_model()
    graph = NetworkMorphismGraph(model)
    graph.to_concat_skip_model(graph.layer_to_id[model.layers[5]], graph.layer_to_id[model.layers[10]])
    graph.to_concat_skip_model(graph.layer_to_id[model.layers[5]], graph.layer_to_id[model.layers[10]])
    graph = NetworkMorphismGraph(graph.produce_model())
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
