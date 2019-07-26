import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import utils
from autokeras.hypermodel import block


class Head(block.Block):
    """Base class for the heads, e.g. classification, regression.

    # Arguments
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        output_shape: Tuple of int(s). Defaults to None. If None, the output shape
            will be inferred from the AutoModel.
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


class ClassificationHead(Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        dropout_rate: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 dropout_rate=None):
        super().__init__(loss=loss, metrics=metrics)
        self.num_classes = num_classes
        self.multi_label = multi_label
        if not self.metrics:
            self.metrics = ['accuracy']
        self.dropout_rate = dropout_rate

    @property
    def loss(self):
        if not self._loss:
            if self.num_classes == 2 or self.multi_label:
                self._loss = 'binary_crossentropy'
            else:
                self._loss = 'categorical_crossentropy'
        return super(ClassificationHead, self).loss

    def build(self, hp, inputs=None):
        if self.num_classes and self.output_shape[-1] != self.num_classes:
            raise ValueError(
                'The data doesn\'t match the num_classes. '
                'Expecting {} but got {}'.format(self.num_classes,
                                                 self.output_shape[-1]))
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0, 0.25, 0.5],
                                                      default=0)

        if dropout_rate > 0:
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        output_node = block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        if self.loss == 'binary_crossentropy':
            output_node = tf.keras.activations.sigmoid(output_node)
        else:
            output_node = tf.keras.layers.Softmax()(output_node)
        return output_node


class RegressionHead(Head):
    """Regression Dense layers.

    Use mean squared error as metrics and loss by default.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        dropout_rate: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 output_dim=None,
                 loss=None,
                 metrics=None,
                 dropout_rate=None):
        super().__init__(loss=loss, metrics=metrics)
        self.output_dim = output_dim
        if not self.metrics:
            self.metrics = ['mean_squared_error']
        self.dropout_rate = dropout_rate

    @property
    def loss(self):
        if not self._loss:
            self._loss = 'mean_squared_error'
        return super(RegressionHead, self).loss

    def build(self, hp, inputs=None):
        if self.output_dim and self.output_shape[-1] != self.output_dim:
            raise ValueError(
                'The data doesn\'t match the output_dim. '
                'Expecting {} but got {}'.format(self.output_dim,
                                                 self.output_shape[-1]))
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0, 0.25, 0.5],
                                                      default=0)

        if dropout_rate > 0:
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        output_node = block.Flatten().build(hp, output_node)
        output_node = tf.keras.layers.Dense(self.output_shape[-1])(output_node)
        return output_node
