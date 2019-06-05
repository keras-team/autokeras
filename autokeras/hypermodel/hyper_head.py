import tensorflow as tf
from autokeras.hypermodel import hyper_block
from autokeras import layer_utils


class HyperHead(hyper_block.HyperBlock):

    def __init__(self, loss=None, metrics=None, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        self.loss = loss
        self.metrics = metrics

    def build(self, hp, inputs=None):
        raise NotImplementedError


class ClassificationHead(HyperHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.metrics:
            self.metrics = [tf.keras.metrics.Accuracy]
        if not self.loss:
            self.loss = tf.keras.losses.categorical_crossentropy

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = hyper_block.Flatten().build(hp, output_node)

        output_node = tf.keras.layers.Dense(self.output_shape)(output_node)
        output_node = tf.keras.layers.Softmax()(output_node)

        # TODO: Add hp.Choice to use sigmoid

        return tf.keras.Model(input_node, output_node)


class RegressionHead(HyperHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.metrics:
            self.metrics = [tf.keras.metrics.mse]
        if not self.loss:
            self.loss = tf.keras.losses.mean_squared_error

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = hyper_block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)

        return output_node
