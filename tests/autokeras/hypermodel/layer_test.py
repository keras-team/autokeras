import numpy as np
import tensorflow as tf

from autokeras.hypermodel import layer as layer_module


def test_feature_encoder_layer():
    data = np.array([['a', 'ab'], ['b', 'bc'], ['a', 'bc']])
    
    input_node = tf.keras.Input(shape=(2,), dtype=tf.string)
    layer = layer_module.FeatureEncodingLayer(['int', 'int'])
    layer.adapt(tf.data.Dataset.from_tensor_slices(data))
    hidden_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(data, np.random.rand(3, 1), epochs=1)

    output = model.predict(data)
    assert output.dtype == np.dtype('float32')
