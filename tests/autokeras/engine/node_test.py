import tensorflow as tf

import autokeras as ak
from autokeras import graph as graph_module


def test_time_series_input_node():
    # TODO. Change test once TimeSeriesBlock is added.
    node = ak.TimeseriesInput(shape=(32,), lookback=2)
    output = node.build()
    assert isinstance(output, tf.Tensor)

    node = graph_module.deserialize(graph_module.serialize(node))
    output = node.build()
    assert isinstance(output, tf.Tensor)
