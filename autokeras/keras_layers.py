import inspect

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.util import nest

CombinerPreprocessingLayer = inspect.getmro(preprocessing.Normalization)[1]
Combiner = inspect.getmro(preprocessing.Normalization()._combiner.__class__)[1]

INT = 'int'
NONE = 'none'
ONE_HOT = 'one-hot'


class MultiColumnCategoricalEncoding(preprocessing.PreprocessingLayer):
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
                self.encoding_layers.append(index_lookup.IndexLookup())
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
                imputed = tf.where(tf.math.is_nan(number),
                                   tf.zeros_like(number),
                                   number)
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
            'encoding': self.encoding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


CUSTOM_OBJECTS = {
    'MultiColumnCategoricalEncoding': MultiColumnCategoricalEncoding,
    'IndexLookup': index_lookup.IndexLookup,
}
