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

import numpy as np
import tensorflow as tf
from official.modeling import activations
from official.nlp.bert import tokenization
from official.nlp.modeling import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import constants

INT = "int"
NONE = "none"
ONE_HOT = "one-hot"


@tf.keras.utils.register_keras_serializable()
class MultiCategoryEncoding(preprocessing.PreprocessingLayer):
    """Encode the categorical features to numerical features.

    # Arguments
        encoding: A list of strings, which has the same number of elements as the
            columns in the structured data. Each of the strings specifies the
            encoding method used for the corresponding column. Use 'int' for
            categorical columns and 'none' for numerical columns.
    """

    # TODO: Support one-hot encoding.
    # TODO: Support frequency encoding.

    def __init__(self, encoding, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.encoding_layers = []
        for encoding in self.encoding:
            if encoding == NONE:
                self.encoding_layers.append(None)
            elif encoding == INT:
                self.encoding_layers.append(preprocessing.StringLookup())
            elif encoding == ONE_HOT:
                self.encoding_layers.append(None)

    def build(self, input_shape):
        for encoding_layer in self.encoding_layers:
            if encoding_layer is not None:
                encoding_layer.build(tf.TensorShape([1]))

    def call(self, inputs):
        input_nodes = nest.flatten(inputs)[0]
        split_inputs = tf.split(input_nodes, [1] * len(self.encoding), axis=-1)
        output_nodes = []
        for input_node, encoding_layer in zip(split_inputs, self.encoding_layers):
            if encoding_layer is None:
                number = tf.strings.to_number(input_node, tf.float32)
                # Replace NaN with 0.
                imputed = tf.where(
                    tf.math.is_nan(number), tf.zeros_like(number), number
                )
                output_nodes.append(imputed)
            else:
                output_nodes.append(tf.cast(encoding_layer(input_node), tf.float32))
        if len(output_nodes) == 1:
            return output_nodes[0]
        return tf.keras.layers.Concatenate()(output_nodes)

    def adapt(self, data):
        for index, encoding_layer in enumerate(self.encoding_layers):
            if encoding_layer is None:
                continue
            data_column = data.map(lambda x: tf.slice(x, [0, index], [-1, 1]))
            encoding_layer.adapt(data_column)

    def get_config(self):
        config = {
            "encoding": self.encoding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO: Remove after KerasNLP is ready.
@tf.keras.utils.register_keras_serializable()
class BertTokenizer(preprocessing.PreprocessingLayer):
    """Vectorization and Encoding the sentences using BERT vocabulary.

    # Arguments
        max_sequence_length: maximum length of the sequences after vectorization.
    """

    def __init__(self, max_sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=constants.BERT_VOCAB_PATH,
            do_lower_case=True,
        )
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update({"max_sequence_length": self.max_sequence_length})
        return config

    def build(self, input_shape):
        self.batch_size = input_shape

    def call(self, inputs):
        input_word_ids = tf.numpy_function(
            func=self.bert_encode, inp=[inputs], Tout=tf.int32
        )
        input_word_ids.set_shape((None, None))
        input_mask = tf.zeros_like(input_word_ids)

        input_type_ids = tf.zeros_like(input_word_ids)

        return input_word_ids, input_mask, input_type_ids

    def encode_sentence(self, s):
        """Encodes a sentence using the BERT tokenizer.

        Tokenizes, and adjusts the sentence length to the maximum sequence
        length. Some important tokens in the BERT tokenizer are:
        [UNK]: 100, [CLS]: 101, [SEP]: 102, [MASK]: 103.

        # Arguments
            s: Tensor. Raw sentence string.
        """
        tokens = list(self.tokenizer.tokenize(s))
        tokens.append("[SEP]")
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tokens)
        return encoded_sentence

    def get_encoded_sentence(self, input_tensor):
        input_array = np.array(input_tensor, dtype=object)
        sentence = tf.ragged.constant(
            [self.encode_sentence(s[0]) for s in input_array]
        )
        return sentence

    def bert_encode(self, input_tensor):
        sentence = self.get_encoded_sentence(input_tensor)
        cls = [self.tokenizer.convert_tokens_to_ids(["[CLS]"])] * sentence.shape[0]
        input_word_ids = tf.concat([cls, sentence], axis=-1).to_tensor()
        if input_word_ids.shape[-1] > self.max_sequence_length:
            input_word_ids = input_word_ids[..., : self.max_sequence_length]

        return input_word_ids


# TODO: Remove after KerasNLP is ready.
@tf.keras.utils.register_keras_serializable()
class BertEncoder(tf.keras.layers.Layer):
    """Cleaned up official.nlp.modeling.networks.TransformerEncoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embedding_width = 768
        dropout_rate = 0.1
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self._embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=30522,
            embedding_width=embedding_width,
            initializer=initializer,
            name="word_embeddings",
        )

        # Always uses dynamic slicing for simplicity.
        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=512,
            name="position_embedding",
        )
        self._type_embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=2,
            embedding_width=embedding_width,
            initializer=initializer,
            use_one_hot=True,
            name="type_embeddings",
        )
        self._add = tf.keras.layers.Add()
        self._layer_norm = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32
        )
        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self._attention_mask = layers.SelfAttentionMask()
        self._transformer_layers = []
        for i in range(12):
            layer = layers.Transformer(
                num_attention_heads=12,
                intermediate_size=3072,
                intermediate_activation=activations.gelu,
                dropout_rate=dropout_rate,
                attention_dropout_rate=0.1,
                output_range=None,
                kernel_initializer=initializer,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        self._lambda = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x[:, 0:1, :], axis=1)
        )
        self._pooler_layer = tf.keras.layers.Dense(
            units=embedding_width,
            activation="tanh",
            kernel_initializer=initializer,
            name="pooler_transform",
        )

    def call(self, inputs):
        word_ids = inputs[0]
        mask = inputs[1]
        type_ids = inputs[2]
        word_embeddings = self._embedding_layer(word_ids)
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(type_ids)

        embeddings = self._add(
            [word_embeddings, position_embeddings, type_embeddings]
        )

        embeddings = self._layer_norm(embeddings)
        embeddings = self._dropout(embeddings)
        data = embeddings
        attention_mask = self._attention_mask([data, mask])
        encoder_outputs = []
        for layer in self._transformer_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = self._lambda(encoder_outputs[-1])

        cls_output = self._pooler_layer(first_token_tensor)

        return cls_output

    def load_pretrained_weights(self):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(constants.BERT_CHECKPOINT_PATH).assert_consumed()
