import kerastuner
import numpy as np


class Encoder(kerastuner.engine.stateful.Stateful):
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

    def get_state(self):
        return {'num_classes': self.num_classes,
                'labels': self._labels,
                'int_to_label': self._int_to_label}

    def set_state(self, state):
        self.num_classes = state['num_classes']
        self._labels = state['labels']
        self._int_to_label = state['int_to_label']


class OneHotEncoder(Encoder):
    """OneHotEncoder to encode and decode the labels.

    This class provides ways to transform data's classification label into vector.

    # Arguments
        num_classes: The number of classes in the classification problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_to_vec = {}

    def get_state(self):
        state = super().get_state()
        state.update({'label_to_vec': self._label_to_vec})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._label_to_vec = state['label_to_vec']

    def fit_with_labels(self, data):
        """Create mapping from label to vector, and vector to label.

        # Arguments
            data: list or numpy.ndarray. The original labels.
        """
        data = np.array(data).flatten()
        self._labels = set(data)
        if not self.num_classes:
            self.num_classes = len(self._labels)
        if self.num_classes < len(self._labels):
            raise ValueError('More classes in data than specified.')
        for index, label in enumerate(self._labels):
            vec = np.array([0] * self.num_classes)
            vec[index] = 1
            self._label_to_vec[label] = vec
            self._int_to_label[index] = label

    def fit_with_one_hot_encoded(self, data):
        """Create mapping from label to vector, and vector to label from one-hot.

        # Arguments
            data: numpy.ndarray. The one-hot encoded labels.
        """
        data = np.array(data)
        if not self.num_classes:
            self.num_classes = data.shape[1]
        self._labels = set(range(self.num_classes))
        for label in self._labels:
            vec = np.array([0] * self.num_classes)
            vec[label] = 1
            self._label_to_vec[label] = vec
            self._int_to_label[label] = label

    def encode(self, data):
        """Get vector for every element in the data array.

        # Arguments
            data: list or numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray. The one-hot encoded labels.
        """
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self._label_to_vec[x], data)))

    def decode(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(list(map(lambda x: self._int_to_label[x],
                                 np.argmax(np.array(data), axis=1)))).reshape(-1, 1)


class LabelEncoder(Encoder):
    """An encoder to encode the labels to integers.

    # Arguments
        num_classes: Int. The number of classes. Defaults to None.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_to_int = {}

    def get_state(self):
        state = super().get_state()
        state.update({'label_to_int': self._label_to_int})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._label_to_int = state['label_to_int']

    def fit_with_labels(self, data):
        """Fit the encoder with all the labels.

        # Arguments
            data: numpy.ndarray. The original labels.
        """
        data = np.array(data).flatten()
        self._labels = set(data)
        if not self.num_classes:
            self.num_classes = len(self._labels)
        if self.num_classes < len(self._labels):
            raise ValueError('More classes in data than specified.')
        for index, label in enumerate(self._labels):
            self._int_to_label[index] = label
            self._label_to_int[label] = index

    def update(self, x):
        if not self.num_classes:
            self.num_classes = 0
        if x not in self._label_to_int:
            self._label_to_int[x] = self.num_classes
            self.num_classes += 1

    def transform(self, x):
        return self._label_to_int[x]

    def encode(self, data):
        """Encode the original labels.

        # Arguments
            data: numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray with shape (n, 1). The encoded labels.
        """
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self._label_to_int[x],
                                 data))).reshape(-1, 1)

    def decode(self, data):
        """Get label for every element in data.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The original labels.
        """
        return np.array(list(map(lambda x: self._int_to_label[int(round(x[0]))],
                                 np.array(data)))).reshape(-1, 1)
