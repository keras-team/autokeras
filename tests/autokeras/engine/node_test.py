import tensorflow as tf

import autokeras as ak
from autokeras import nodes


def test_time_series_input_node():
    # TODO. Change test once TimeSeriesBlock is added.
    node = ak.TimeseriesInput(shape=(32,), lookback=2)
    output = node.build()
    assert isinstance(output, tf.Tensor)

    node = nodes.deserialize(nodes.serialize(node))
    output = node.build()
    assert isinstance(output, tf.Tensor)
