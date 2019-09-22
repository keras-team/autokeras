import numpy as np
import pandas as pd
import tensorflow as tf


class Node(object):
    """The nodes in a network connecting the blocks."""
    # TODO: Implement get_config() and set_config(), so that the entire graph can
    # be saved.

    def __init__(self, shape=None):
        super().__init__()
        self.in_blocks = []
        self.out_blocks = []
        self.shape = shape

    def add_in_block(self, hypermodel):
        self.in_blocks.append(hypermodel)

    def add_out_block(self, hypermodel):
        self.out_blocks.append(hypermodel)

    def build(self):
        return tf.keras.Input(shape=self.shape)

    def clear_edges(self):
        self.in_blocks = []
        self.out_blocks = []


class TextNode(Node):
    pass


class Input(Node):

    def fit(self, y):
        """Record any information needed by transform."""
        pass

    def transform(self, x):
        """Transform x into a compatible type (tf.data.Dataset)."""
        if isinstance(x, tf.data.Dataset):
            return x
        if isinstance(x, np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return tf.data.Dataset.from_tensor_slices(x)
        raise TypeError('Unsupported type {type} for '
                        '{name}.'.format(type=type(x),
                                         name=self.__class__.__name__))


class ImageInput(Input):

    def transform(self, x):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = x.reshape(-1, 1)
        return super().transform(x)


class TextInput(Input, TextNode):
    pass


class StructuredDataInput(Input):
    def __init__(self, column_names=None, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types

    def fit(self, x):
        if not isinstance(x, (pd.DataFrame, np.ndarray)):
            raise TypeError('Unsupported type {type} for '
                            '{name}.'.format(type=type(x),
                                             name=self.__class__.__name__))

        # Extract column_names from pd.DataFrame.
        if isinstance(x, pd.DataFrame) and self.column_names is None:
            self.column_names = list(x.columns)
            # column_types is provided by user
            if (self.column_types and
                not all([column_name in self.column_names
                         for column_name in self.column_types])):
                raise ValueError('Column_names and column_types are mismatched.')

        # Generate column_names.
        if self.column_names is None:
            if self.column_types:
                raise ValueError('Column names must be specified.')
            self.column_names = [index for index in range(x.shape[1])]

        # Check if column_names has the correct length.
        if len(self.column_names) != x.shape[1]:
            raise ValueError('The length of column_names and data are mismatched.')

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            # convert x,y,validation_data to tf.Dataset
            x = tf.data.Dataset.from_tensor_slices(
                x.values.astype(np.unicode))
        return super().transform(x)


class TimeSeriesInput(Input):
    pass
