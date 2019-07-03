import tensorflow as tf
from autokeras.hypermodel import hyper_block
from autokeras import utils


class HyperHead(hyper_block.HyperBlock):

    def __init__(self, loss=None, metrics=None, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        self.loss = loss
        self.metrics = metrics

    def build(self, hp, inputs=None):
        raise NotImplementedError


class ClassificationHead(HyperHead):

    def __init__(self, binary=False, **kwargs):
        super().__init__(**kwargs)
        self.binary = binary
        if not self.metrics:
            self.metrics = ['accuracy']
        if not self.loss:
            if binary:
                self.loss = 'binary_crossentropy'
                self.output_shape = (1,)
            else:
                self.loss = 'categorical_crossentropy'

    def build(self, hp, inputs=None):
        input_node = utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = hyper_block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        if self.binary:
            output_node = tf.keras.activations.sigmoid(output_node)
        else:
            output_node = tf.keras.layers.Softmax()(output_node)
        return output_node


class RegressionHead(HyperHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.metrics:
            self.metrics = ['mean_squared_error']
        if not self.loss:
            self.loss = 'mean_squared_error'

    def build(self, hp, inputs=None):
        input_node = utils.format_inputs(inputs, self.name, num=1)[0]
        output_node = input_node
        output_node = hyper_block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        return output_node
