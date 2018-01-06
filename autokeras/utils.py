import os

from keras import backend
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, MaxPooling1D
from tensorflow import Dimension

from autokeras import constant
from autokeras.constant import CONV_FUNC_LIST


def is_conv_layer(layer):
    """Return whether the layer is convolution layer"""
    return isinstance(layer, tuple(CONV_FUNC_LIST))


def get_conv_layer_func(n_dim):
    """Return convolution function based on the dimension"""
    conv_layer_functions = [Conv1D, Conv2D, Conv3D]
    if n_dim > 3:
        raise ValueError('The input dimension is too high.')
    if n_dim < 1:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 1]


class ModelTrainer:
    """A class that is used to train model

    This class can train a model with dataset and will not stop until getting minimum loss

    Attributes:
        model: the model that will be trained
        x_train: the input train data
        y_train: the input train data labels
        x_test: the input test data
        y_test: the input test data labels
        verbose: verbosity mode
        training_losses: a list to store all losses during training
        minimum_loss: the minimum loss during training
        _no_improvement_count: the number of iterations that don't improve the result
    """
    def __init__(self, model, x_train, y_train, x_test, y_test, verbose):
        """Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose"""
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0

    def _converged(self, loss):
        """Return whether the training is converged"""
        self.training_losses.append(loss)
        if loss > (self.minimum_loss - constant.MIN_LOSS_DEC):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        if loss < self.minimum_loss:
            self.minimum_loss = loss

        return self._no_improvement_count > constant.MAX_NO_IMPROVEMENT_NUM

    def train_model(self):
        """Train the model with dataset and return the minimum_loss"""
        self.training_losses = []
        self._no_improvement_count = 0
        self.minimum_loss = float('inf')
        for _ in range(constant.MAX_ITER_NUM):
            self.model.fit(self.x_train, self.y_train,
                           batch_size=min(self.x_train.shape[0], 200),
                           epochs=constant.EPOCHS_EACH,
                           verbose=self.verbose)
            loss, _ = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)
            if self._converged(loss):
                break
        return self.minimum_loss


def copy_layer(layer, input_shape=None):
    """Return a copied layer"""
    if input_shape is None:
        input_shape = layer.input_shape
    layer_config = layer.get_config()
    layer_config.pop('name', None)
    new_layer = layer.__class__.from_config(layer_config)
    new_layer.build(input_shape)
    new_layer.set_weights(layer.get_weights())
    return new_layer


def extract_config(network):
    """Return configuration of one model"""
    return network.get_config()


def ensure_dir(directory):
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    """Create path if it does not exist"""
    ensure_dir(os.path.dirname(path))


def has_file(path):
    """Return whether the path has a file"""
    return os.path.exists(path)


def reset_weights(model):
    """Reset weights with a new model"""
    session = backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def get_layer_size(layer):
    """Return the size of layer"""
    if is_conv_layer(layer):
        return layer.filters
    return layer.units


def get_int_tuple(temp_shape):
    """Return the input shape of temp_shape in the form of tuple"""
    input_shape = []
    for i in temp_shape:
        if isinstance(i, Dimension):
            input_shape.append(i.value)
        else:
            input_shape.append(i)
    return tuple(input_shape)


def is_pooling_layer(layer):
    return isinstance(layer, (MaxPooling1D, MaxPooling2D, MaxPooling3D))
