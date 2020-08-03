# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import tensorflow as tf

from autokeras import keras_layers as layer_module


def get_data():
    train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    test = np.array([["a", "ab", 2.1], ["x", "bc", 1.0], ["a", "bc", "nan"]])
    y = np.random.rand(3, 1)
    return train, test, y


def test_multi_column_categorical_encoding(tmp_path):
    x_train, x_test, y_train = get_data()
    input_node = tf.keras.Input(shape=(3,), dtype=tf.string)
    layer = layer_module.MultiCategoryEncoding(
        [layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    hidden_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1, activation="sigmoid")(hidden_node)
    model = tf.keras.Model(input_node, output_node)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    tf.data.Dataset.zip(
        (
            (tf.data.Dataset.from_tensor_slices(x_train).batch(32),),
            (tf.data.Dataset.from_tensor_slices(np.random.rand(3, 1)).batch(32),),
        )
    )
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))

    model.fit(x_train, y_train, epochs=1)

    model2 = tf.keras.Model(input_node, hidden_node)
    result = model2.predict(x_train)
    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0

    output = model2.predict(x_test)
    assert output.dtype == np.dtype("float32")


def build_model():
    input_node = tf.keras.Input(shape=(3,), dtype=tf.string)
    layer = layer_module.MultiCategoryEncoding(
        encoding=[layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    output_node = layer(input_node)
    output_node = tf.keras.layers.Dense(1)(output_node)
    return tf.keras.Model(input_node, output_node), layer


def test_model_save(tmp_path):
    x_train, x_test, y_train = get_data()
    model, layer = build_model()
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=1, verbose=False)

    model.save(os.path.join(tmp_path, "model"))
    model2 = tf.keras.models.load_model(os.path.join(tmp_path, "model"))

    assert np.array_equal(model.predict(x_train), model2.predict(x_train))


def test_weight_save(tmp_path):
    x_train, x_test, y_train = get_data()
    model, layer = build_model()
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=1, verbose=False)
    model.save_weights(os.path.join(tmp_path, "checkpoint"))

    model2, _ = build_model()
    model2.load_weights(os.path.join(tmp_path, "checkpoint"))

    assert np.array_equal(model.predict(x_train), model2.predict(x_train))


def test_init_multi_one_hot_encode():
    layer_module.MultiCategoryEncoding(
        encoding=[layer_module.ONE_HOT, layer_module.INT, layer_module.NONE]
    )
    # TODO: add more content when it is implemented


def test_call_multi_with_single_column():
    layer = layer_module.MultiCategoryEncoding(encoding=[layer_module.INT])

    assert layer(np.array([["a"], ["b"], ["a"]])).shape == (3, 1)
