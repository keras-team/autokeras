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
import re
from typing import List

import numpy as np
import tensorflow as tf
from official.modeling import activations
from official.nlp.bert import tokenization
from official.nlp.modeling import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import constants
from autokeras.utils import data_utils

INT = "int"
NONE = "none"
ONE_HOT = "one-hot"


@tf.keras.utils.register_keras_serializable()
class CastToFloat32(preprocessing.PreprocessingLayer):
    def call(self, inputs):
        return data_utils.cast_to_float32(inputs)


@tf.keras.utils.register_keras_serializable()
class ExpandLastDim(preprocessing.PreprocessingLayer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)


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

    def __init__(self, encoding: List[str], **kwargs):
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
                number = data_utils.cast_to_float32(input_node)
                # Replace NaN with 0.
                imputed = tf.where(
                    tf.math.is_nan(number), tf.zeros_like(number), number
                )
                output_nodes.append(imputed)
            else:
                output_nodes.append(
                    data_utils.cast_to_float32(
                        encoding_layer(data_utils.cast_to_string(input_node))
                    )
                )
        if len(output_nodes) == 1:
            return output_nodes[0]
        return tf.keras.layers.Concatenate()(output_nodes)

    def adapt(self, data):
        for index, encoding_layer in enumerate(self.encoding_layers):
            if encoding_layer is None:
                continue
            data_column = data.map(lambda x: tf.slice(x, [0, index], [-1, 1]))
            encoding_layer.adapt(data_column.map(data_utils.cast_to_string))

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

    def __init__(self, max_sequence_length: int, **kwargs):
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


@tf.keras.utils.register_keras_serializable()
class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.

    Instead we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay_rate=0.0,
        include_in_weight_decay=None,
        exclude_from_weight_decay=None,
        name="AdamWeightDecay",
        **kwargs
    ):
        super(AdamWeightDecay, self).__init__(
            learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs
        )
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(
            var_device, var_dtype, apply_state
        )
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate
                * var
                * apply_state[(var.device, var.dtype.base_dtype)][
                    "weight_decay_rate"
                ],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(
        self, grads_and_vars, name=None, experimental_aggregate_gradients=True
    ):
        grads, tvars = list(zip(*grads_and_vars))
        if experimental_aggregate_gradients:
            # when experimental_aggregate_gradients = False, apply_gradients() no
            # longer implicitly allreduce gradients, users manually allreduce
            # gradient and passed the allreduced grads_and_vars. For now, the
            # clip_by_global_norm will be moved to before the explicit allreduce to
            # keep the math the same as TF 1 and pre TF 2.2 implementation.
            (grads, _) = tf.clip_by_global_norm(
                grads, clip_norm=1.0
            )  # pragma: no cover
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients,
        )

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}  # pragma: no cover

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(
                var_device, var_dtype
            )  # pragma: no cover
            apply_state[(var_device, var_dtype)] = coefficients  # pragma: no cover

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(
                grad, var, **kwargs
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state
        )  # pragma: no cover
        decay = self._decay_weights_op(var, lr_t, apply_state)  # pragma: no cover
        with tf.control_dependencies([decay]):  # pragma: no cover
            return super(
                AdamWeightDecay, self  # pragma: no cover
            )._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update(
            {
                "weight_decay_rate": self.weight_decay_rate,
            }
        )
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


@tf.keras.utils.register_keras_serializable()
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate,
        decay_schedule_fn,
        warmup_steps,
        power=1.0,
        name=None,
    ):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
