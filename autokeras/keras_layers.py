import collections
import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

CombinerPreprocessingLayer = inspect.getmro(preprocessing.Normalization)[1]
Combiner = inspect.getmro(preprocessing.Normalization()._combiner.__class__)[1]

INT = 'int'
NONE = 'none'
ONE_HOT = 'one-hot'


class CategoricalEncoding(CombinerPreprocessingLayer):
    """Encode the categorical features to numerical features.

    # Arguments
        encoding: A list of strings, which has the same number of elements as the
            columns in the structured data. Each of the strings specifies the
            encoding method used for the corresponding column. Use 'int' for
            categorical columns and 'none' for numerical columns.
    """

    # TODO: Support one-hot encoding.

    def __init__(self, encoding, **kwargs):
        super().__init__(
            combiner=CategoricalEncodingCombiner(encoding),
            **kwargs)
        self.encoding = encoding
        self.tables = {
            str(index): tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=-3,
                empty_key='-2',
                deleted_key='-1'
            )
            for index, method in enumerate(self.encoding)
            if method in [INT, ONE_HOT]
        }

        for key, table in self.tables.items():
            tracked_table = self._add_trackable(table, trainable=False)
            tracked_table.shape = tf.TensorShape((0,))

    def _set_state_variables(self, updates):
        for key, vocab in updates.items():
            self.tables[key].insert(
                np.array(vocab, dtype=np.str),
                np.arange(len(vocab))
            )

    def call(self, inputs):
        inputs = nest.flatten(inputs)[0]
        outputs = []
        for index in range(len(self.encoding)):
            col = tf.slice(inputs, [0, index], [-1, 1])
            if self.encoding[index] in [INT, ONE_HOT]:
                col = self.tables[str(index)].lookup(col)
                col = tf.cast(col, tf.float32)
            else:
                col = tf.strings.to_number(col, tf.float32)
            outputs.append(col)
        outputs = tf.concat(outputs, axis=-1)
        outputs.set_shape(inputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_spec):
        return input_spec

    def get_config(self):
        config = {'encoding': self.encoding}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CategoricalEncodingCombiner(Combiner):

    def __init__(self, encoding):
        self.encoding = encoding

    def compute(self, values, accumulator=None):
        if accumulator is None:
            accumulator = collections.defaultdict(set)
        for line in K.get_value(values):
            for index, value in enumerate(line):
                if self.encoding[index] in [INT, ONE_HOT]:
                    accumulator[str(index)].add(value)
        return accumulator

    def merge(self, accumulators):
        base_accumulator = collections.defaultdict(set)
        for accumulator in accumulators:
            for index, method in enumerate(self.encoding):
                if method in [INT, ONE_HOT]:
                    base_accumulator[index] = base_accumulator[index].union(
                        accumulator[index])
        return base_accumulator

    def extract(self, accumulator):
        return {
            key: list(value)
            for key, value in accumulator.items()
        }

    def restore(self, output):
        return {
            key: set(value)
            for key, value in output
        }

    def serialize(self, accumulator):
        pass

    def deserialize(self, encoded_accumulator):
        pass


class LookbackPreprocessing(CombinerPreprocessingLayer):
    """Transform 2-D time series data to 3-D to be consumed by RNN.

    # Arguments
        lookback: Int. The range of history steps to consider for each prediction.
            For example, if lookback=n, the data in the range of [i - n, i - 1]
            is used to predict the value of step i. If unspecified, it will be tuned
            automatically.
    """

    def __init__(self, lookback, **kwargs):
        super().__init__(
            combiner=LookbackPreprocessingCombiner(lookback),
            **kwargs)
        self.lookback = lookback

    def _set_state_variables(self, updates):
        for key, value in updates.items():
            self.output_len = value - self.lookback + 1

    def call(self, inputs):
        # TODO Handle inputs that are smaller than lookback.
        inputs = nest.flatten(inputs)[0]
        input_shape = K.shape(inputs)
        input_len = input_shape[0]
        pad = tf.zeros(shape=(self.lookback-1, self.lookback, input_shape[1]))
        outputs = tf.map_fn(
            lambda i: inputs[i:i+self.lookback],
            tf.range(input_len - self.lookback + 1),
            dtype=tf.float32)
        outputs.set_shape(self.compute_output_shape(K.int_shape(inputs)))
        outputs = tf.concat([pad, outputs], 0)
        # outputs = nest.flatten(inputs)[0]
        # outputs.set_shape(compute_output_shape(inputs.shape))
        return outputs

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape([input_shape[0],
                                         self.lookback,
                                         input_shape[1]])

    def compute_output_signature(self, input_spec):
        return input_spec

    def get_config(self):
        config = {'lookback': self.lookback}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LookbackPreprocessingCombiner(Combiner):

    def __init__(self, lookback, **kwargs):
        self.lookback = lookback

    # TODO. Determine what to keep in accumulator. Not necessary though.
    def compute(self, values, accumulator=None):
        if accumulator is None:
            accumulator = collections.defaultdict(int)
        for line in K.get_value(values):
            accumulator["input_len"] += 1
        return accumulator

    def merge(self, accumulators):
        base_accumulator = collections.defaultdict(int)
        for accumulator in accumulators:
            base_accumulator["input_len"] += accumulator["input_len"]
        return base_accumulator

    def extract(self, accumulator):
        return {
            key: value
            for key, value in accumulator.items()
        }

    def restore(self, output):
        return {
            key: value
            for key, value in output
        }

    def serialize(self, accumulator):
        pass

    def deserialize(self, encoded_accumulator):
        pass


class Sigmoid(tf.keras.layers.Layer):
    """Sigmoid activation function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.activations.sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


CUSTOM_OBJECTS = {
    'CategoricalEncoding': CategoricalEncoding,
    'LookbackPreprocessing': LookbackPreprocessing,
    'Sigmoid': Sigmoid,
}
