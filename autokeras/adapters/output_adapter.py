import numpy as np
import pandas as pd
import tensorflow as tf

from autokeras import encoders
from autokeras.engine import adapter as adapter_module
from autokeras.utils import data_utils


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

    def __init__(self,
                 num_classes=None,
                 multi_label=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_encoder = None
        self.multi_label = multi_label

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': encoders.serialize(self.label_encoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj.label_encoder = encoders.deserialize(config['encoder'])

    def fit_before_convert(self, dataset):
        """Fit the encoder."""
        # If in tf.data.Dataset, must be encoded already.
        if isinstance(dataset, tf.data.Dataset):
            return

        # Convert the data to np.ndarray.
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values.reshape(-1, 1)

        # If encoded.
        # TODO: support raw string labels for multi-label.
        if len(dataset.flatten()) != len(dataset):
            if self.num_classes:
                self._check_data_shape(dataset.shape[1:])
            return

        # Fit encoder.
        labels = set(dataset.flatten())
        if len(labels) < 2:
            raise ValueError('Expect the target data for {name} to have '
                             'at least 2 classes, but got {num_classes}.'
                             .format(name=self.name, num_classes=self.num_classes))
        if len(labels) == 2 and not self.multi_label:
            self.label_encoder = encoders.LabelEncoder()
        else:
            self.label_encoder = encoders.OneHotEncoder()
        self.label_encoder.fit(dataset)

    def convert_to_dataset(self, dataset):
        if self.label_encoder:
            dataset = self.label_encoder.encode(dataset)
        return super().convert_to_dataset(dataset)

    def fit(self, dataset):
        super().fit(dataset)
        shape = tuple(data_utils.dataset_shape(dataset).as_list()[1:])
        # Infer the num_classes.
        if not self.num_classes:
            # Single column with 0s and 1s.
            if shape == (1,):
                self.num_classes = 2
            else:
                self.num_classes = shape[0]
            return

        # Compute expected shape from num_classes.
        if self.num_classes == 2 and not self.multi_label:
            expected = (1,)
        else:
            expected = (self.num_classes,)

        # Check shape equals expected shape.
        if shape != expected:
            raise ValueError('Expect the target data for {name} to have '
                             'shape {expected}, but got {actual}.'
                             .format(name=self.name, expected=expected,
                                     actual=shape))

    def postprocess(self, y):
        if self.multi_label:
            y[y < 0.5] = 0
            y[y > 0.5] = 1
        if self.label_encoder:
            y = self.label_encoder.decode(y)
        return y


class RegressionHeadAdapter(HeadAdapter):
    pass


class SegmentationHeadAdapter(ClassificationHeadAdapter):
    pass
