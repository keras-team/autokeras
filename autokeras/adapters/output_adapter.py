import numpy as np
import pandas as pd
import tensorflow as tf

from autokeras import encoder
from autokeras.engine import adapter as adapter_module


class HeadAdapter(adapter_module.Adapter):

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def check(self, dataset):
        supported_types = (tf.data.Dataset, np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(dataset, supported_types):
            raise TypeError('Expect the target data of {name} to be tf.data.Dataset,'
                            ' np.ndarray, pd.DataFrame or pd.Series, but got {type}.'
                            .format(name=self.name, type=type(dataset)))

    def convert_to_dataset(self, dataset):
        if isinstance(dataset, np.ndarray):
            if len(dataset.shape) == 1:
                dataset = dataset.reshape(-1, 1)
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values.reshape(-1, 1)
        return super().convert_to_dataset(dataset)

    def postprocess(self, y):
        """Postprocess the output of the Keras Model."""
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'name': self.name,
        })
        return config


class ClassificationHeadAdapter(HeadAdapter):

    def __init__(self, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_encoder = None

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': encoder.serialize(self.label_encoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj.label_encoder = encoder.deserialize(config['encoder'])

    def fit_before_convert(self, dataset):
        # If in tf.data.Dataset, must be encoded already.
        if isinstance(dataset, tf.data.Dataset):
            if not self.num_classes:
                shape = dataset.take(1).shape[1]
                if shape == 1:
                    self.num_classes = 2
                else:
                    self.num_classes = shape
            return
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values.reshape(-1, 1)
        # Not label.
        if len(dataset.flatten()) != len(dataset):
            self.num_classes = dataset.shape[1]
            return
        labels = set(dataset.flatten())
        if self.num_classes is None:
            self.num_classes = len(labels)
        if self.num_classes == 2:
            self.label_encoder = encoder.LabelEncoder()
        elif self.num_classes > 2:
            self.label_encoder = encoder.OneHotEncoder()
        elif self.num_classes < 2:
            raise ValueError('Expect the target data for {name} to have '
                             'at least 2 classes, but got {num_classes}.'
                             .format(name=self.name, num_classes=self.num_classes))
        self.label_encoder.fit_with_labels(dataset)

    def convert_to_dataset(self, dataset):
        if self.label_encoder:
            dataset = self.label_encoder.encode(dataset)
        return super().convert_to_dataset(dataset)

    def postprocess(self, y):
        if self.label_encoder:
            y = self.label_encoder.decode(y)
        return y


class RegressionHeadAdapter(HeadAdapter):
    pass
