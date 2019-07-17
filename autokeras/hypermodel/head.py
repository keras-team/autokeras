import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import block


class HyperHead(block.HyperBlock):
    """Base class for the heads, e.g. classification, regression.

    Attributes:
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            infered from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be infered from the AutoModel.
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be infered from the AutoModel.
    """

    def __init__(self, loss=None, metrics=None, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape
        self._loss = loss
        self.metrics = metrics

    def build(self, hp, inputs=None):
        raise NotImplementedError

    @property
    def loss(self):
        return self._loss


class ClassificationHead(HyperHead):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classificaiton and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    Args:
        num_classes: Int. Defaults to None.
        multi_label: Boolean. Defaults to False.
    """

    def __init__(self, num_classes=None, multi_label=False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.multi_label = multi_label
        if not self.metrics:
            self.metrics = ['accuracy']

    @property
    def loss(self):
        if not self._loss:
            if self.num_classes == 2 or self.multi_label:
                self._loss = 'binary_crossentropy'
            else:
                self._loss = 'categorical_crossentropy'
        return super(ClassificationHead, self).loss

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        if self.loss == 'binary_crossentropy':
            output_node = tf.keras.activations.sigmoid(output_node)
        else:
            output_node = tf.keras.layers.Softmax()(output_node)
        return output_node


class RegressionHead(HyperHead):
    """Regression Dense layers.

    Use mean squared error as metrics and loss by default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.metrics:
            self.metrics = ['mean_squared_error']

    @property
    def loss(self):
        if not self._loss:
            self._loss = 'mean_squared_error'
        return super(RegressionHead, self).loss

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        return output_node
