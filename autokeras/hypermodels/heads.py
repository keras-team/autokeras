from tensorflow.keras import layers
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import keras_layers
from autokeras import utils
from autokeras.engine import head as head_module
from autokeras.hypermodels import reduction


class ClassificationHead(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout_rate: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(loss=loss,
                         metrics=metrics,
                         **kwargs)
        self.num_classes = num_classes
        self.multi_label = multi_label
        if not self.metrics:
            self.metrics = ['accuracy']
        self.dropout_rate = dropout_rate
        self.set_loss()

    def set_loss(self):
        if not self.num_classes:
            return
        if not self.loss:
            if self.num_classes == 2 or self.multi_label:
                self.loss = 'binary_crossentropy'
            elif self.num_classes > 2:
                self.loss = 'categorical_crossentropy'

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'multi_label': self.multi_label,
            'dropout_rate': self.dropout_rate})
        return config

    def build(self, hp, inputs=None):
        if self.num_classes:
            expected = self.num_classes if self.num_classes > 2 else 1
            if self.output_shape[-1] != expected:
                raise ValueError(
                    'The data doesn\'t match the expected shape. '
                    'Expecting {} but got {}'.format(expected,
                                                     self.output_shape[-1]))
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        # Reduce the tensor to a vector.
        if len(output_node.shape) > 2:
            output_node = reduction.SpatialReduction().build(hp, output_node)

        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0)

        if dropout_rate > 0:
            output_node = layers.Dropout(dropout_rate)(output_node)
        output_node = layers.Dense(self.output_shape[-1])(output_node)
        if self.loss == 'binary_crossentropy':
            output_node = keras_layers.Sigmoid(name=self.name)(output_node)
        else:
            output_node = layers.Softmax(name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ClassificationHeadAdapter(name=self.name)

    def config_from_adapter(self, adapter):
        super().config_from_adapter(adapter)
        self.num_classes = adapter.num_classes
        self.set_loss()


class RegressionHead(head_module.Head):
    """Regression Dense layers.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be single-column or multi-column. The
    values should all be numerical.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `mean_squared_error`.
        metrics: A list of Keras metrics. Defaults to use `mean_squared_error`.
        dropout_rate: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self,
                 output_dim=None,
                 loss='mean_squared_error',
                 metrics=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(loss=loss,
                         metrics=metrics,
                         **kwargs)
        self.output_dim = output_dim
        if not self.metrics:
            self.metrics = ['mean_squared_error']
        self.loss = loss
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate})
        return config

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
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        if dropout_rate > 0:
            output_node = layers.Dropout(dropout_rate)(output_node)
        output_node = reduction.Flatten().build(hp, output_node)
        output_node = layers.Dense(self.output_shape[-1],
                                   name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.RegressionHeadAdapter(name=self.name)
