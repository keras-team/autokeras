from autokeras.graph import *
from tests.common import get_conv_model, get_conv_data, get_add_skip_model, get_conv_dense_model, get_pooling_model


def test_graph():
    graph = Graph(get_conv_model())
    assert graph.n_nodes == 7


def test_conv_deeper():
    model = get_conv_model()
    graph = Graph(model)
    new_model = graph.to_conv_deeper_model(model.layers[4], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert abs(np.sum(output1 - output2)) < 0.2


def test_dense_deeper():
    model = get_conv_dense_model()
    graph = Graph(model)
    new_model = graph.to_dense_deeper_model(model.layers[5])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(output1 - output2) == 0


def test_conv_wider():
    model = get_add_skip_model()
    graph = Graph(model)
    new_model = graph.to_wider_model(model.layers[7], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert abs(np.sum(output1 - output2)) < 1e-4


def test_dense_wider():
    model = get_add_skip_model()
    graph = Graph(model)
    new_model = graph.to_wider_model(model.layers[-2], 3)
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert abs(np.sum(output1 - output2)) < 1e-4


def test_skip_add():
    model = get_conv_model()
    graph = Graph(model)
    new_model = graph.to_add_skip_model(model.layers[1], model.layers[4])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_add_over_pooling():
    model = get_pooling_model()
    graph = Graph(model)
    new_model = graph.to_add_skip_model(model.layers[4], model.layers[11])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.array_equal(output1, output2)


def test_skip_concatenate():
    model = get_add_skip_model()
    graph = Graph(model)
    new_model = graph.to_concat_skip_model(model.layers[4], model.layers[4])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert abs(np.sum(output1 - output2)) < 1e-4


def test_skip_concat_over_pooling():
    model = get_pooling_model()
    graph = Graph(model)
    new_model = graph.to_concat_skip_model(model.layers[4], model.layers[11])
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert abs(np.sum(output1 - output2)) < 1e-4


def test_copy_model():
    model = get_add_skip_model()
    new_model = Graph(model).produce_model()
    input_data = get_conv_data()

    output1 = model.predict_on_batch(input_data).flatten()
    output2 = new_model.predict_on_batch(input_data).flatten()

    assert np.sum(output1 - output2) == 0
