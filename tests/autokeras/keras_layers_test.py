import numpy as np
import tensorflow as tf

from autokeras import keras_layers as layer_module
from autokeras.engine import tuner


def test_feature_encoder_layer():
    data = np.array([['a', 'ab'], ['b', 'bc'], ['a', 'bc']])

    input_node = tf.keras.Input(shape=(2,), dtype=tf.string)
    layer = layer_module.CategoricalEncoding([
        layer_module.INT,
        layer_module.INT,
    ])
    hidden_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    dataset = tf.data.Dataset.zip((
        (tf.data.Dataset.from_tensor_slices(data).batch(32),),
        (tf.data.Dataset.from_tensor_slices(np.random.rand(3, 1)).batch(32),),
    ))
    tuner.AutoTuner._adapt_model(model, dataset)

    model.fit(data, np.random.rand(3, 1), epochs=1)

    output = model.predict(data)

    model2 = tf.keras.Model(input_node, hidden_node)
    result = model2.predict(data)
    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert output.dtype == np.dtype('float32')


def test_lookback_preprocessor_layer():
    data = np.array([[0.12, 0.12], [0.13, 0.13], [0.14, 0.14], [0.15, 0.15],
                     [0.16, 0.16]])

    predict = np.random.rand(5, 1)
    lookback = 2

    input_node = tf.keras.Input(shape=(2,), dtype=tf.float32)
    layer = layer_module.LookbackPreprocessing(lookback)
    hidden_node = layer(input_node)
    lstm_node = tf.keras.layers.LSTM(10, activation='relu')(hidden_node)
    output_node = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    dataset = tf.data.Dataset.zip((
        (tf.data.Dataset.from_tensor_slices(data).batch(32),),
        (tf.data.Dataset.from_tensor_slices(predict).batch(32),),
    ))
    tuner.AutoTuner._adapt_model(model, dataset)

    model.fit(data, predict, epochs=1)

    output = model.predict(data)

    model2 = tf.keras.Model(input_node, hidden_node)
    result = model2.predict(data)
    assert result.shape == (5, 2, 2)
    assert np.sum(result[0]) == 0.0
    assert output.dtype == np.dtype('float32')
