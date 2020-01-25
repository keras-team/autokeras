import collections
import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import backend as K
from tensorflow.python.util import nest

CombinerPreprocessingLayer = inspect.getmro(preprocessing.Normalization)[1]
Combiner = inspect.getmro(preprocessing.Normalization()._combiner.__class__)[1]


class FeatureEncodingLayer(CombinerPreprocessingLayer):

    INT = 'int'
    ONE_HOT = 'one-hot'
    # TODO: Support one-hot encoding.

    def __init__(self, encoding, **kwargs):
        super().__init__(
            combiner=FeatureEncodingCombiner(encoding),
            **kwargs)
        self.encoding = encoding
        self.tables = {
            index: tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=-3,
                empty_key='-2',
                deleted_key='-1'
            )
            for index, method in enumerate(self.encoding)
            if method in [self.INT, self.ONE_HOT]
        }

        for key, table in self.tables.items():
            tracked_table = self._add_trackable(table, trainable=False)
            tracked_table.shape = tf.TensorShape((0,))

    def _set_state_variables(self, updates):
        for key, vocab in updates.items():
            self.tables[int(key)].insert(
                np.array(vocab, dtype=np.str), 
                np.arange(len(vocab))
            )

    def call(self, inputs):
        inputs = nest.flatten(inputs)[0]
        outputs = []
        for index in range(len(self.encoding)):
            col = tf.slice(inputs, [0, index], [-1, 1])
            if self.encoding[index] in [self.INT, self.ONE_HOT]:
                col = self.tables[index].lookup(col)
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


class FeatureEncodingCombiner(Combiner):

    def __init__(self, encoding):
        self.encoding = encoding

    def compute(self, values, accumulator=None):
        if accumulator is None:
            accumulator = collections.defaultdict(set)
        for index, value in enumerate(K.get_value(values)):
            if self.encoding[index] in [self.INT, self.ONE_HOT]:
                accumulator[index].add(value)
        return accumulator

    def merge(self, accumulators):
        base_accumulator = collections.defaultdict(set)
        for accumulator in accumulators:
            for index, method in enumerate(self.encoding):
                if method in [self.INT, self.ONE_HOT]:
                    print(accumulator[index])
                    base_accumulator[index] = base_accumulator[index].union(
                        accumulator[index])
        return base_accumulator

    def extract(self, accumulator):
        return {
            str(key): list(value)
            for key, value in accumulator.items()
        }

    def restore(self, output):
        return {
            int(key): set(value)
            for key, value in output
        }

    def serialize(self, accumulator):
        pass

    def deserialize(self, encoded_accumulator):
        pass
