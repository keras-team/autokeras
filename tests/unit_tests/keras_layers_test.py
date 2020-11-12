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


def test_multi_cat_encode_strings_correctly(tmp_path):
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    layer = layer_module.MultiCategoryEncoding(
        [layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)

    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    for data in dataset.map(layer):
        result = data

    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0
    assert result.dtype == tf.float32


def test_model_save_load_output_same(tmp_path):
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    layer = layer_module.MultiCategoryEncoding(
        encoding=[layer_module.INT, layer_module.INT, layer_module.NONE]
    )
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))

    model = tf.keras.Sequential([tf.keras.Input(shape=(3,), dtype=tf.string), layer])
    model.save(os.path.join(tmp_path, "model"))
    model2 = tf.keras.models.load_model(os.path.join(tmp_path, "model"))

    assert np.array_equal(model.predict(x_train), model2.predict(x_train))


def test_init_multi_one_hot_encode():
    layer_module.MultiCategoryEncoding(
        encoding=[layer_module.ONE_HOT, layer_module.INT, layer_module.NONE]
    )
    # TODO: add more content when it is implemented


def test_call_multi_with_single_column_return_right_shape():
    layer = layer_module.MultiCategoryEncoding(encoding=[layer_module.INT])

    assert layer(np.array([["a"], ["b"], ["a"]])).shape == (3, 1)


def get_text_data():
    train = np.array(
        [
            ["This is a test example"],
            ["This is another text example"],
            ["Is this another example?"],
            [""],
            ["Is this a long long long long long long example?"],
        ],
        dtype=np.str,
    )
    test = np.array(
        [
            ["This is a test example"],
            ["This is another text example"],
            ["Is this another example?"],
        ],
        dtype=np.str,
    )
    y = np.random.rand(3, 1)
    return train, test, y


def test_bert_tokenizer_output_correct_shape(tmp_path):
    x_train, x_test, y_train = get_text_data()
    max_sequence_length = 8
    token_layer = layer_module.BertTokenizer(max_sequence_length=max_sequence_length)
    output = token_layer(x_train)
    assert output[0].shape == (x_train.shape[0], max_sequence_length)
    assert output[1].shape == (x_train.shape[0], max_sequence_length)
    assert output[2].shape == (x_train.shape[0], max_sequence_length)


def test_bert_tokenizer_save_and_load(tmp_path):
    x_train, x_test, y_train = get_text_data()
    max_sequence_length = 8
    layer = layer_module.BertTokenizer(max_sequence_length=max_sequence_length)

    input_node = tf.keras.Input(shape=(1,), dtype=tf.string)
    output_node = layer(input_node)
    model = tf.keras.Model(input_node, output_node)
    model.save(os.path.join(tmp_path, "model"))
    model2 = tf.keras.models.load_model(os.path.join(tmp_path, "model"))

    assert np.array_equal(model.predict(x_train), model2.predict(x_train))


def test_transformer_encoder_save_and_load(tmp_path):
    layer = layer_module.BertEncoder()
    inputs = [
        tf.keras.Input(shape=(500,), dtype=tf.int64),
        tf.keras.Input(shape=(500,), dtype=tf.int64),
        tf.keras.Input(shape=(500,), dtype=tf.int64),
    ]
    model = tf.keras.Model(inputs, layer(inputs))
    model.save(os.path.join(tmp_path, "model"))
    tf.keras.models.load_model(os.path.join(tmp_path, "model"))


def test_adam_weight_decay(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(10,))])
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        end_learning_rate=0.0,
    )
    lr_schedule = layer_module.WarmUp(
        initial_learning_rate=0.1,
        decay_schedule_fn=lr_schedule,
        warmup_steps=10,
    )
    optimizer = layer_module.AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )
    model.compile(loss="mse", optimizer=optimizer)
    model.fit(np.random.rand(100, 10), np.random.rand(100, 10), epochs=2)
    model.save(os.path.join(tmp_path, "model"))
