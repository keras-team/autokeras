def test_set_hp():
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    graph = ak.hypermodel.graph.GraphHyperModel(
        input_node,
        output_node,
        directory=tmp_dir,
        max_trials=1)
    graph.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    result = graph.predict(x_train)

    assert result.shape == (100, 1)
