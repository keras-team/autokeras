import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras.engine import adapter as adapter_module
from autokeras.utils import data_utils

CATEGORICAL = 'categorical'
NUMERICAL = 'numerical'


class InputAdapter(adapter_module.Adapter):

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to Input to be numpy.ndarray or '
                            'tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError('Expect the data to Input to be numerical, but got '
                            '{type}.'.format(type=x.dtype))


class ImageInputAdapter(adapter_module.Adapter):

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to ImageInput to be numpy.ndarray or '
                            'tf.data.Dataset, but got {type}.'.format(type=type(x)))
        if isinstance(x, np.ndarray) and x.ndim not in [3, 4]:
            raise ValueError('Expect the data to ImageInput to have 3 or 4 '
                             'dimensions, but got input shape {shape} with {ndim} '
                             'dimensions'.format(shape=x.shape, ndim=x.ndim))
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError('Expect the data to ImageInput to be numerical, but got '
                            '{type}.'.format(type=x.dtype))

    def convert_to_dataset(self, x):
        if isinstance(x, np.ndarray):
            # TODO: expand the dims after converting to Dataset.
            if x.ndim == 3:
                x = np.expand_dims(x, axis=3)
            x = x.astype(np.float32)
        return super().convert_to_dataset(x)


class TextInputAdapter(adapter_module.Adapter):

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to TextInput to be numpy.ndarray or '
                            'tf.data.Dataset, but got {type}.'.format(type=type(x)))

        if isinstance(x, np.ndarray) and x.ndim != 1:
            raise ValueError('Expect the data to TextInput to have 1 dimension, but '
                             'got input shape {shape} with {ndim} dimensions'.format(
                                 shape=x.shape,
                                 ndim=x.ndim))
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.character):
            raise TypeError('Expect the data to TextInput to be strings, but got '
                            '{type}.'.format(type=x.dtype))

    def convert_to_dataset(self, x):
        x = super().convert_to_dataset(x)
        shape = data_utils.dataset_shape(x)
        if len(shape) == 1:
            x = x.map(lambda a: tf.reshape(a, [-1, 1]))
        return x


class StructuredDataInputAdapter(adapter_module.Adapter):

    def __init__(self, column_names=None, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        # Variables for inferring column types.
        self.count_nan = None
        self.count_numerical = None
        self.count_categorical = None
        self.count_unique_numerical = []
        self.num_col = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'count_nan': self.count_nan,
            'count_numerical': self.count_numerical,
            'count_categorical': self.count_categorical,
            'count_unique_numerical': self.count_unique_numerical,
            'num_col': self.num_col
        })
        return config

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj.count_nan = config['count_nan']
        obj.count_numerical = config['count_numerical']
        obj.count_categorical = config['count_categorical']
        obj.count_unique_numerical = config['count_unique_numerical']
        obj.num_col = config['num_col']

    def check(self, x):
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Unsupported type {type} for '
                            '{name}.'.format(type=type(x),
                                             name=self.__class__.__name__))

        # Extract column_names from pd.DataFrame.
        if isinstance(x, pd.DataFrame) and self.column_names is None:
            self.column_names = list(x.columns)
            # column_types is provided by user
            if self.column_types:
                for column_name in self.column_types:
                    if column_name not in self.column_names:
                        raise ValueError('Column_names and column_types are '
                                         'mismatched. Cannot find column name '
                                         '{name} in the data.'.format(
                                             name=column_name))

        if self.column_names is None:
            if self.column_types:
                raise ValueError('Column names must be specified.')

    def convert_to_dataset(self, x):
        if isinstance(x, pd.DataFrame):
            # Convert x, y, validation_data to tf.Dataset.
            x = x.values.astype(np.unicode)
        if isinstance(x, np.ndarray):
            x = x.astype(np.unicode)
        return super().convert_to_dataset(x)

    def fit(self, dataset):
        super().fit(dataset)
        for x in dataset:
            self.update(x)
        self.infer_column_types()

    def update(self, x):
        # Calculate the statistics.
        x = nest.flatten(x)[0].numpy()
        for instance in x:
            self._update_instance(instance)

    def _update_instance(self, x):
        if self.num_col is None:
            self.num_col = len(x)
            self.count_nan = np.zeros(self.num_col)
            self.count_numerical = np.zeros(self.num_col)
            self.count_categorical = np.zeros(self.num_col)
            for i in range(len(x)):
                self.count_unique_numerical.append({})
        for i in range(self.num_col):
            x[i] = x[i].decode('utf-8')
            if x[i] == 'nan':
                self.count_nan[i] += 1
            elif x[i] == 'True':
                self.count_categorical[i] += 1
            elif x[i] == 'False':
                self.count_categorical[i] += 1
            else:
                try:
                    tmp_num = float(x[i])
                    self.count_numerical[i] += 1
                    if tmp_num not in self.count_unique_numerical[i]:
                        self.count_unique_numerical[i][tmp_num] = 1
                    else:
                        self.count_unique_numerical[i][tmp_num] += 1
                except ValueError:
                    self.count_categorical[i] += 1

    def infer_column_types(self):
        column_types = {}

        if self.column_names is None:
            # Generate column names.
            self.column_names = [index for index in range(self.num_col)]
        # Check if column_names has the correct length.
        elif len(self.column_names) != self.num_col:
            raise ValueError('Expect column_names to have length {expect} '
                             'but got {actual}.'.format(
                                 expect=self.num_col,
                                 actual=len(self.column_names)))

        for i in range(self.num_col):
            if self.count_categorical[i] > 0:
                column_types[self.column_names[i]] = CATEGORICAL
            elif len(self.count_unique_numerical[i])/self.count_numerical[i] < 0.05:
                column_types[self.column_names[i]] = CATEGORICAL
            else:
                column_types[self.column_names[i]] = NUMERICAL
        # Partial column_types is provided.
        if self.column_types is None:
            self.column_types = {}
        for key, value in column_types.items():
            if key not in self.column_types:
                self.column_types[key] = value


class TimeseriesInputAdapter(adapter_module.Adapter):

    def __init__(self,
                 lookback=None,
                 column_names=None,
                 column_types=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.column_names = column_names
        self.column_types = column_types

    def get_config(self):
        config = super().get_config()
        config.update({
            'lookback': self.lookback,
            'column_names': self.column_names,
            'column_types': self.column_types
        })
        return config

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data in TimeseriesInput to be numpy.ndarray'
                            ' or tf.data.Dataset or pd.DataFrame, but got {type}.'.
                            format(type=type(x)))

        if isinstance(x, np.ndarray) and x.ndim != 2:
            raise ValueError('Expect the data in TimeseriesInput to have 2 dimension'
                             ', but got input shape {shape} with {ndim} '
                             'dimensions'.format(
                                 shape=x.shape,
                                 ndim=x.ndim))

        # Extract column_names from pd.DataFrame.
        if isinstance(x, pd.DataFrame) and self.column_names is None:
            self.column_names = list(x.columns)
            # column_types is provided by user
            if self.column_types:
                for column_name in self.column_types:
                    if column_name not in self.column_names:
                        raise ValueError('Column_names and column_types are '
                                         'mismatched. Cannot find column name '
                                         '{name} in the data.'.format(
                                             name=column_name))

        # Generate column_names.
        if self.column_names is None:
            if self.column_types:
                raise ValueError('Column names must be specified.')
            self.column_names = [index for index in range(x.shape[1])]

        # Check if column_names has the correct length.
        if len(self.column_names) != x.shape[1]:
            raise ValueError('Expect column_names to have length {expect} '
                             'but got {actual}.'.format(
                                 expect=x.shape[1],
                                 actual=len(self.column_names)))

    def convert_to_dataset(self, x):
        if isinstance(x, pd.DataFrame):
            # Convert x, y, validation_data to tf.Dataset.
            x = x.values.astype(np.float32)
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.window(self.lookback, shift=1, drop_remainder=True)
        final_data = []
        for window in x:
            final_data.append([elems.numpy() for elems in window])
        final_data = tf.data.Dataset.from_tensor_slices(final_data)
        return super().convert_to_dataset(final_data)
