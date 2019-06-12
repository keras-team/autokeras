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

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        if not self.metrics:
            self.metrics = [tf.keras.metrics.categorical_accuracy]
        if not self.loss:
            self.loss = tf.keras.losses.categorical_crossentropy
        if num_classes:
            self.output_shape = (num_classes,)

    def build(self, hp, inputs=None):
        input_node = layer_utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        if len(self.output_shape) == 1:
            output_node = hyper_block.Flatten().build(hp, output_node)
            output_node = tf.keras.layers.Dense(
                self.output_shape[0])(output_node)
            output_node = tf.keras.layers.Softmax()(output_node)
            return output_node

        # TODO: Add hp.Choice to use sigmoid

        return hyper_block.Reshape(self.output_shape).build(hp, output_node)


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
        if len(self.output_shape) == 1:
            output_node = hyper_block.Flatten().build(hp, output_node)
            output_node = tf.keras.layers.Dense(
                self.output_shape[-1])(output_node)
            return output_node
        return hyper_block.Reshape(self.output_shape).build(hp, output_node)
