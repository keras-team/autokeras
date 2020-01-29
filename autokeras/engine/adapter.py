import numpy as np
import tensorflow as tf

from autokeras import utils
from autokeras.engine import serializable


class Adapter(serializable.Serializable):
    """Adpat the input and output format for Keras Model."""

    def __init__(self, shape=None, **kwargs):
        self.shape = shape

    def check(self, dataset):
        pass

    def convert_to_dataset(self, x):
        if isinstance(x, tf.data.Dataset):
            return x
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            return tf.data.Dataset.from_tensor_slices(x)

    def fit(self, dataset):
        self.record_dataset_shape(dataset)

    def fit_before_convert(self, dataset):
        pass

    def fit_transform(self, dataset):
        self.check(dataset)
        self.fit_before_convert(dataset)
        dataset = self.convert_to_dataset(dataset)
        self.fit(dataset)
        return dataset

    def record_dataset_shape(self, dataset):
        self.shape = utils.dataset_shape(dataset)

    def transform(self, dataset):
        self.check(dataset)
        return self.convert_to_dataset(dataset)

    def get_config(self):
        return {'shape': self.shape}
