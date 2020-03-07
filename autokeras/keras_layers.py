import collections
import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
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
    'Sigmoid': Sigmoid,
    'Normalization': preprocessing.Normalization
}
