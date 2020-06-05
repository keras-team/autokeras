import tensorflow as tf

from autokeras import adapters
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='nodes')


class Input(node_module.Node, io_hypermodel.IOHyperModel):
    """Input node for tensor data.

    The data should be numpy.ndarray or tf.data.Dataset.
    """

    def build(self):
        return tf.keras.Input(shape=self.shape, dtype=tf.float32)

    def get_adapter(self):
        return adapters.InputAdapter()

    def config_from_adapter(self, adapter):
        self.shape = adapter.shape


class ImageInput(Input):
    """Input node for image data.

    The input data should be numpy.ndarray or tf.data.Dataset. The shape of the data
    should be should be (samples, width, height) or
    (samples, width, height, channels).
    """

    def get_adapter(self):
        return adapters.ImageInputAdapter()


class TextInput(Input):
    """Input node for text data.

    The input data should be numpy.ndarray or tf.data.Dataset. The data should be
    one-dimensional. Each element in the data should be a string which is a full
    sentence.
    """

    def build(self):
        return tf.keras.Input(shape=self.shape, dtype=tf.string)

    def get_adapter(self):
        return adapters.TextInputAdapter()


class StructuredDataInput(Input):
    """Input node for structured data.

    The input data should be numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
    The data should be two-dimensional with numerical or categorical values.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
    """

    def __init__(self, column_names=None, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types

    def build(self):
        return tf.keras.Input(shape=self.shape, dtype=tf.string)

    def get_config(self):
        config = super().get_config()
        config.update({
            'column_names': self.column_names,
            'column_types': self.column_types,
        })
        return config

    def get_adapter(self):
        return adapters.StructuredDataInputAdapter(
            self.column_names,
            self.column_types
        )

    def config_from_adapter(self, adapter):
        super().config_from_adapter(adapter)
        self.column_names = adapter.column_names
        self.column_types = adapter.column_types


class TimeseriesInput(Input):
    """Input node for timeseries data.

    # Arguments
        lookback: Int. The range of history steps to consider for each prediction.
            For example, if lookback=n, the data in the range of [i - n, i - 1]
            is used to predict the value of step i. If unspecified, it will be tuned
            automatically.
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
    """

    def __init__(self,
                 lookback=None,
                 column_names=None,
                 column_types=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.column_names = column_names
        self.column_types = column_types

    def build(self):
        if len(self.shape) == 1:
            self.shape = (self.lookback, self.shape[0],)
        return tf.keras.Input(shape=self.shape, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'lookback': self.lookback,
            'column_names': self.column_names,
            'column_types': self.column_types
        })
        return config

    def get_adapter(self):
        return adapters.TimeseriesInputAdapter(lookback=self.lookback,
                                               column_names=self.column_names,
                                               column_types=self.column_types)

    def config_from_adapter(self, adapter):
        super().config_from_adapter(adapter)
        self.lookback = adapter.lookback
        self.column_names = adapter.column_names
        self.column_types = adapter.column_types
