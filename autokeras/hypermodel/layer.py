import collections
import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

CombinerPreprocessingLayer = inspect.getmro(preprocessing.Normalization)[1]
Combiner = inspect.getmro(preprocessing.Normalization()._combiner.__class__)[1]

INT = 'int'
ONE_HOT = 'one-hot'


class FeatureEncodingLayer(CombinerPreprocessingLayer):

    def __init__(self, encoding, **kwargs):
        super().__init__(
            combiner=FeatureEncodingCombiner(encoding),
            **kwargs)
        self.encoding = encoding
        self.categorical_values = {}

    def build(self, input_shape):
        super().build(input_shape)
        for index in range(input_shape[0]):
            if self.encoding[index] in [INT, ONE_HOT]:
                categorical_value_dict = self._add_state_variable(
                    name=str(index),
                    shape=(None,),
                    dtype=tf.string,
                    initializer=lambda shape, dtype=None: [],
                )
                self.categorical_values[index] = categorical_value_dict

    def call(self, inputs):
        value_to_encoding = {}
        for key, values in self.categorical_values.items():
            mapping = {}
            value_to_encoding[key] = mapping
            for encoding, value in enumerate(values.numpy()):
                mapping[value] = encoding

        outputs = []
        for element in inputs.numpy():
            row = []
            for index, value in enumerate(element):
                if self.encoding[index] == INT:
                    print(value_to_encoding[index].keys())
                    row.append(value_to_encoding[index][value])
                else:
                    row.append(float(value))
            outputs.append(row)
        return np.array(outputs).astype(np.float32)

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
        for index, value in enumerate(values.numpy()):
            if self.encoding[index] in [INT, ONE_HOT]:
                accumulator[index].add(value)
        return accumulator

    def merge(self, accumulators):
        base_accumulator = collections.defaultdict(set)
        for accumulator in accumulators:
            for index, method in enumerate(self.encoding):
                if method in [INT, ONE_HOT]:
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
