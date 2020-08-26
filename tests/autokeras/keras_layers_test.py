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
import official.nlp.bert.tokenization
import tensorflow as tf
from tensorflow.keras import losses

from autokeras import keras_layers as layer_module
from autokeras.applications import BERT


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


def get_text_data():
    train = np.array(
        [
            "This is a test example",
            "This is another text example",
            "Is this another example?",
            "",
            "Is this a long long long long long long example?"
        ]
    )
    test = np.array(
        [
            "This is a test example",
            "This is another text example",
            "Is this another example?",
        ]
    )
    y = np.random.rand(3, 1)
    return train, test, y


class bert_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(bert_layer, self).__init__()
        self.bert_encoder = BERT()

    def call(self, inputs):
        bert_input = {
            "input_word_ids": inputs[0],
            "input_mask": inputs[1],
            "input_type_ids": inputs[2],
        }
        output = self.bert_encoder(bert_input, training=True,)
        return output[1]


def test_text_vectorization_with_tokenizer(tmp_path):
    x_train, x_test, y_train = get_data()
    gs_folder_bert = (
        "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
    )
    tokenizer = official.nlp.bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(gs_folder_bert, "vocab.txt"), do_lower_case=True
    )
    token_layer = layer_module.TextVectorizationWithTokenizer(
        tokenizer=tokenizer, max_seq_len=8
    )
    output = token_layer(x_train)
    print(output.shape)
    assert output.dtype == np.dtype("int32")


# def test_text_vectorization_with_tokenizer(tmp_path):
#     x_train, x_test, y_train = get_data()
#     gs_folder_bert = (
#         "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
#     )
#     tokenizer = official.nlp.bert.tokenization.FullTokenizer(
#         vocab_file=os.path.join(gs_folder_bert, "vocab.txt"), do_lower_case=True
#     )
#     input_node = tf.keras.Input(shape=(1,), dtype=tf.string)
#     token_layer = layer_module.TextVectorizationWithTokenizer(
#         tokenizer=tokenizer, max_seq_len=16
#     )
#     hidden_node = token_layer(input_node)
#     bert_output = bert_layer()(hidden_node)
#     output_node = tf.keras.layers.Dense(2)(bert_output)
#     model = tf.keras.Model(input_node, output_node)
#     model.compile(
#         loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam"
#     )
#     tf.data.Dataset.zip(
#         (
#             (tf.data.Dataset.from_tensor_slices(x_train).batch(32),),
#             (tf.data.Dataset.from_tensor_slices(np.random.rand(3, 1)).batch(32),),
#         )
#     )
#     model.fit(x_train, y_train, epochs=1)
#
#     model2 = tf.keras.Model(input_node, hidden_node)
#     result = model2.predict(x_train)
#     print("SHAPE of RESULT: ", result.shape, result)
#
#     output = model2.predict(x_test)
#     assert output.dtype == np.dtype("int32")
