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
from autokeras.applications import bert


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
            ["Is this a long long long long long long example?"]
        ],
        dtype=np.str
    )
    test = np.array(
        [
            ["This is a test example"],
            ["This is another text example"],
            ["Is this another example?"],
        ],
        dtype=np.str
    )
    y = np.random.rand(3, 1)
    return train, test, y


class bert_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(bert_layer, self).__init__()
        self.bert_encoder = bert.BERT()

    def call(self, inputs):
        bert_input = {
            "input_word_ids": inputs[0],
            "input_mask": inputs[1],
            "input_type_ids": inputs[2],
        }
        output = self.bert_encoder(bert_input, training=True,)
        return output[1]


def test_text_vectorization_with_tokenizer(tmp_path):
    x_train, x_test, y_train = get_text_data()
    tokenizer = official.nlp.bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(bert.GS_FOLDER_BERT, "vocab.txt"),
        do_lower_case=True
    )
    max_seq_len = 8
    token_layer = layer_module.TextVectorizationWithTokenizer(
        tokenizer=tokenizer, max_seq_len=max_seq_len
    )
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)

    for data in dataset.map(token_layer):
        output = data
    # output = token_layer(x_train)
    print(output.shape)
    # # assert output.dtype == np.dtype("int32")
    assert output.shape == (3, x_train.shape[0], max_seq_len)


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
