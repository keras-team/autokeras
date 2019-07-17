import numpy as np
import tensorflow as tf
from sklearn import feature_selection
from sklearn.feature_extraction import text
from tensorflow.python.util import nest

from autokeras import const
from autokeras.hypermodel import hyper_block as hb_module


class HyperPreprocessor(hb_module.HyperBlock):
    """Hyper preprocessing block base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hp = None

    def build(self, hp, inputs=None):
        """Build into part of a Keras Model.

        Since they are for preprocess data before feeding into the Keras Model,
        they are not part of the Keras Model. They only pass the inputs
        directly to outputs.
        """
        return inputs

    def set_hp(self, hp):
        """Set Hyperparameters for the Preprocessor.

        Since the `update` and `transform` function are all for single training
        instances instead of the entire dataset, the Hyperparameters needs to be
        set in advance of call them.

        Args:
            hp: Hyperparameters. The hyperparameters for tuning the preprocessor.
        """
        self._hp = hp

    def update(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        Args:
            x: EagerTensor. A single instance in the training dataset.
        """
        raise NotImplementedError

    def transform(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        Args:
            x: EagerTensor. A single instance in the training dataset.

        Returns:
            A transformed instanced which can be converted to a tf.Tensor.
        """
        raise NotImplementedError

    def output_types(self):
        """The output types of the transformed data, e.g. tf.int64.

        The output types are required by tf.py_function, which is used for transform
        the dataset into a new one with a map function.

        Returns:
            A tuple of data types.
        """
        raise NotImplementedError

    def output_shape(self):
        """The output shape of the transformed data.

        The output shape is needed to build the Keras Model from the AutoModel.
        The output shape of the preprocessor is the input shape of the Keras Model.

        Returns:
            A tuple of ints or a TensorShape.
        """
        raise NotImplementedError

    def finalize(self):
        """Training process of the preprocessor after update with all instances."""
        pass


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into
    vector.

    Attributes:
        data: The input data
        num_classes: The number of classes in the classification problem.
        labels: The number of labels.
        label_to_vec: Mapping from label to vector.
        int_to_label: Mapping from int to label.
    """

    def __init__(self):
        """Initialize a OneHotEncoder"""
        self.data = None
        self.num_classes = 0
        self.labels = None
        self.label_to_vec = {}
        self.int_to_label = {}

    def fit(self, data):
        """Create mapping from label to vector, and vector to label."""
        data = np.array(data).flatten()
        self.labels = set(data)
        self.num_classes = len(self.labels)
        for index, label in enumerate(self.labels):
            vec = np.array([0] * self.num_classes)
            vec[index] = 1
            self.label_to_vec[label] = vec
            self.int_to_label[index] = label

    def transform(self, data):
        """Get vector for every element in the data array."""
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.flatten()
        return np.array(list(map(lambda x: self.label_to_vec[x], data)))

    def inverse_transform(self, data):
        """Get label for every element in data."""
        return np.array(list(map(lambda x: self.int_to_label[x],
                                 np.argmax(np.array(data), axis=1))))


class Normalize(HyperPreprocessor):
    """ Perform basic image transformation and augmentation.

    # Attributes
        mean: Tensor. The mean value. Shape: (data last dimension length,)
        std: Tensor. The standard deviation. Shape is the same as mean.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sum = 0
        self.square_sum = 0
        self.count = 0
        self.mean = None
        self.std = None
        self._shape = None

    def update(self, x):
        x = nest.flatten(x)[0].numpy()
        self.sum += x
        self.square_sum += np.square(x)
        self.count += 1
        self._shape = x.shape

    def finalize(self):
        axis = tuple(range(len(self._shape) - 1))
        self.mean = np.mean(self.sum / self.count, axis=axis)
        square_mean = np.mean(self.square_sum / self.count, axis=axis)
        self.std = np.sqrt(square_mean - np.square(self.mean))

    def transform(self, x):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Tensorflow Dataset. The data to be transformed.

        # Returns
            A DataLoader instance.
        """
        x = nest.flatten(x)[0]
        return (x - self.mean) / self.std

    def output_types(self):
        return tf.float64,

    def output_shape(self):
        return self.mean.shape


class TextToIntSequence(HyperPreprocessor):
    """Convert raw texts to sequences of word indices."""

    def __init__(self, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self._max_len = 0
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)

    def update(self, x):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        self._tokenizer.fit_on_texts([sentence])
        sequence = self._tokenizer.texts_to_sequences([sentence])[0]
        if self.max_len is None:
            self._max_len = max(self._max_len, len(sequence))

    def transform(self, x):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        sequence = self._tokenizer.texts_to_sequences(sentence)[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            self.max_len or self._max_len)
        return sequence

    def output_types(self):
        return tf.int64,

    def output_shape(self):
        return self.max_len or self._max_len,


class TextToNgramVector(HyperPreprocessor):
    """Convert raw texts to n-gram vectors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorizer = text.TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',
            min_df=2)
        self.selector = None
        self.labels = None
        self._max_features = const.Constant.VOCABULARY_SIZE
        self._vectorizer.max_features = self._max_features
        self._texts = []
        self._shape = None

    def update(self, x):
        # TODO: Implement a sequential version fit for both
        #  TfidfVectorizer and SelectKBest
        self._texts.append(nest.flatten(x)[0].numpy().decode('utf-8'))

    def finalize(self):
        self._texts = np.array(self._texts)
        self._vectorizer.fit(self._texts)
        data = self._vectorizer.transform(self._texts)
        self._shape = data.shape[1:]
        if self.labels:
            self.selector = feature_selection.SelectKBest(
                feature_selection.f_classif,
                k=min(self._max_features, data.shape[1]))
            self.selector.fit(data, self.labels)

    def transform(self, x):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        data = self._vectorizer.transform([sentence]).toarray()
        if self.selector:
            data = self.selector.transform(data).astype('float32')
        return data[0]

    def output_types(self):
        return tf.float64,

    def output_shape(self):
        return self._shape
