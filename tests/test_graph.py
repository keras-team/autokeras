from autokeras.graph import *
from tests.common import get_conv_model, get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model, \
    get_concat_skip_model


def test_graph():
    graph = Graph(get_conv_model())
    assert graph.n_nodes == 7


def test_conv_deeper():
    model = get_conv_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_conv_deeper_model(model.layers[4], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_dense_deeper():
    model = get_conv_dense_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_dense_deeper_model(model.layers[5])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_conv_wider():
    model = get_add_skip_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_wider_model(model.layers[7], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_dense_wider():
    model = get_add_skip_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_wider_model(model.layers[-2], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()
    print(output1, output2)
    print(np.sum(np.abs(output1 - output2)))

    assert np.sum(np.abs(output1 - output2)) < 1e-4


def test_skip_add():
    model = get_conv_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_add_skip_model(model.layers[1], model.layers[4])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_add_over_pooling():
    model = get_pooling_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_add_skip_model(model.layers[4], model.layers[11])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_concatenate():
    model = get_add_skip_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_concat_skip_model(model.layers[4], model.layers[4])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_skip_concat_over_pooling():
    model = get_pooling_model()
    graph = NetworkMorphismGraph(model)
    new_model = graph.to_concat_skip_model(model.layers[4], model.layers[11])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(np.abs(output1 - output2)) < 1e-1


def test_copy_model():
    model = get_add_skip_model()
    new_model = NetworkMorphismGraph(model).produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(output1 - output2) == 0


def test_extract_descriptor_add():
    model = get_add_skip_model()
    descriptor = Graph(model).extract_descriptor()
    assert descriptor.n_conv == 4
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.ADD_CONNECT), (3, 4, NetworkDescriptor.ADD_CONNECT)]


def test_extract_descriptor_concat():
    model = get_concat_skip_model()
    descriptor = Graph(model).extract_descriptor()
    assert descriptor.n_conv == 4
    assert descriptor.n_dense == 2
    assert descriptor.skip_connections == [(2, 3, NetworkDescriptor.CONCAT_CONNECT),
                                           (3, 4, NetworkDescriptor.CONCAT_CONNECT)]
