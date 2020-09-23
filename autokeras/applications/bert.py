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

import json
import os

from official.nlp.bert import configs
from official.modeling import tf_utils
import tensorflow as tf

from autokeras import constants
from autokeras import keras_layers



def BertEncoder():
    config_dict = json.loads(tf.io.gfile.GFile(constants.BERT_CONFIG_PATH).read())

    bert_config = configs.BertConfig.from_dict(config_dict)

    bert_encoder = keras_layers.BertEncoder()
        # vocab_size=bert_config.vocab_size,
        # hidden_size=bert_config.hidden_size,
        # num_layers=bert_config.num_hidden_layers,
        # num_attention_heads=bert_config.num_attention_heads,
        # intermediate_size=bert_config.intermediate_size,
        # activation=tf_utils.get_activation(bert_config.hidden_act),
        # dropout_rate=bert_config.hidden_dropout_prob,
        # attention_dropout_rate=bert_config.attention_probs_dropout_prob,
        # sequence_length=None,
        # max_sequence_length=bert_config.max_position_embeddings,
        # type_vocab_size=bert_config.type_vocab_size,
        # embedding_width=bert_config.embedding_size,
        # initializer=tf.keras.initializers.TruncatedNormal(
            # stddev=bert_config.initializer_range))



    return bert_encoder


def OriginBert():
    config_dict = json.loads(tf.io.gfile.GFile(constants.BERT_CONFIG_PATH).read())

    bert_config = configs.BertConfig.from_dict(config_dict)

    bert_encoder = keras_layers.networks.TransformerEncoder(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        activation=tf_utils.get_activation(bert_config.hidden_act),
        dropout_rate=bert_config.hidden_dropout_prob,
        attention_dropout_rate=bert_config.attention_probs_dropout_prob,
        sequence_length=None,
        max_sequence_length=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        embedding_width=bert_config.embedding_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range))

    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(
        constants.BERT_CHECKPOINT_PATH 
    ) .assert_consumed()

    return bert_encoder
