import numpy as np
import tensorflow as tf

from autokeras.hypermodel import layer as layer_module


def test_feature_encoder_layer():
    data = tf.data.Dataset.from_tensor_slices(np.array(
        [['a', 'ab'], ['b', 'bc'], ['a', 'bc']]))
    batch = next(iter(data.batch(3)))
    layer = layer_module.FeatureEncodingLayer(['int', 'int'])
    layer.build((2,))
    layer.adapt(data)
    output = layer(batch)
    assert output.dtype == np.dtype('float32')
