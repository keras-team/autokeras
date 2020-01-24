import numpy as np
import tensorflow as tf

from autokeras.hypermodel import layer as layer_module


def test_feature_encoder_layer():
    data = np.array([['a', 'ab'], ['b', 'bc'], ['a', 'bc']])
    
    input_node = tf.keras.Input(shape=(2,), dtype=tf.string)
    layer = layer_module.FeatureEncodingLayer(['int', 'int'])
    hidden_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(data, np.random.rand(3, 1), epochs=1)

    model2 = tf.keras.Model(input_node, hidden_node)
    output = model2.predict(data)
    assert np.array_equal(output, np.array([[1, 1], [0, 0], [1, 0]]))
    assert output.dtype == np.dtype('float32')
