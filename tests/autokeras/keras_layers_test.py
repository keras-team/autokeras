import numpy as np
import tensorflow as tf

from autokeras import keras_layers as layer_module


def test_feature_encoder_layer(tmp_path):
    data = np.array([['a', 'ab', 2.1], ['b', 'bc', 1.0], ['a', 'bc', 'nan']])
    data2 = np.array([['a', 'ab', 2.1], ['x', 'bc', 1.0], ['a', 'bc', 'nan']])

    input_node = tf.keras.Input(shape=(3,), dtype=tf.string)
    layer = layer_module.MultiColumnCategoricalEncoding([
        layer_module.INT,
        layer_module.INT,
        layer_module.NONE,
    ])
    hidden_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    tf.data.Dataset.zip((
        (tf.data.Dataset.from_tensor_slices(data).batch(32),),
        (tf.data.Dataset.from_tensor_slices(np.random.rand(3, 1)).batch(32),),
    ))
    layer.adapt(tf.data.Dataset.from_tensor_slices(data).batch(32))

    model.fit(data, np.random.rand(3, 1), epochs=1)

    output = model.predict(data)

    model2 = tf.keras.Model(input_node, hidden_node)
    result = model2.predict(data)
    model2.predict(data2)
    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0
    assert output.dtype == np.dtype('float32')
