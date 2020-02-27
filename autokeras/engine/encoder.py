from autokeras.engine import serializable


class Encoder(serializable.Serializable):
    """Base class for encoders of the prediction targets.

    # Arguments
        num_classes: Int. The number of classes. Defaults to None.
    """

    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self._labels = None
        self._int_to_label = {}

    def fit_with_labels(self, data):
        """Fit the encoder with all the labels.

        # Arguments
            data: numpy.ndarray. The original labels.
        """
        raise NotImplementedError

    def encode(self, data):
        """Encode the original labels.

        # Arguments
            data: numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray. The encoded labels.
        """
        raise NotImplementedError

    def decode(self, data):
        """Decode the encoded labels to original labels.

        # Arguments
            data: numpy.ndarray. The encoded labels.

        # Returns
            numpy.ndarray. The original labels.
        """
        raise NotImplementedError

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'labels': self._labels,
            'int_to_label': self._int_to_label,
        }

    @classmethod
    def from_config(cls, config):
        obj = super().from_config(config)
        obj._labels = config['labels']
        obj._int_to_label = config['int_to_label']
