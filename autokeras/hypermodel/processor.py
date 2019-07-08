import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.feature_extraction import text
from sklearn import feature_selection

from autokeras import const
from autokeras.hypermodel import hyper_block as hb_module


class HyperPreprocessor(hb_module.HyperBlock):

    def build(self, hp, inputs=None):
        return inputs

    def fit_transform(self, hp, inputs):
        self.fit(hp, inputs)
        return self.transform(hp, inputs)

    def fit(self, hp, inputs):
        raise NotImplementedError

    def transform(self, hp, inputs):
        raise NotImplementedError


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into
    vector.

    # Attributes
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
        self.mean = None
        self.std = None

    def fit(self, hp, inputs):
        data = np.array(list(tfds.as_numpy(inputs)))
        axis = tuple(range(len(data.shape) - 1))
        self.mean = np.mean(data,
                            axis=axis,
                            keepdims=True).flatten()
        self.std = np.std(data,
                          axis=axis,
                          keepdims=True).flatten()

    def transform(self, hp, inputs):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Tensorflow Dataset. The data to be transformed.

        # Returns
            A DataLoader instance.
        """

        # channel-wise normalize the image
        data = np.array(list(tfds.as_numpy(inputs)))
        data = (data - self.mean) / self.std
        return tf.data.Dataset.from_tensor_slices(data)


class TextToSequenceVector(HyperPreprocessor):
    def __init__(self, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)

    def fit(self, hp, inputs):
        texts = np.array(list(tfds.as_numpy(inputs))).astype(np.str)
        self._tokenizer.fit_on_texts(texts)
        sequences = self._tokenizer.texts_to_sequences(texts)
        if not self.max_len:
            self.max_len = len(max(sequences, key=len))

    def transform(self, hp, inputs):
        texts = np.array(list(tfds.as_numpy(inputs))).astype(np.str)
        sequences = self._tokenizer.texts_to_sequences(texts)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                  self.max_len)
        return tf.data.Dataset.from_tensor_slices(sequences)


class TextToNgramVector(HyperPreprocessor):
    def __init__(self, labels=None, **kwargs):
        super().__init__(**kwargs)
        self._vectorizer = text.TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',
            min_df=2)
        self.selector = None
        self.labels = labels
        self._max_features = const.Constant.VOCABULARY_SIZE
        if not labels:
            self._vectorizer.max_features = self._max_features

    def fit(self, hp, inputs):
        texts = np.array(list(tfds.as_numpy(inputs))).astype(np.str)
        self._vectorizer.fit(texts)
        data = self._vectorizer.transform(texts)
        if self.labels:
            self.selector = feature_selection.SelectKBest(
                feature_selection.f_classif,
                k=min(self._max_features, data.shape[1]))
            self.selector.fit(data, self.labels)

    def transform(self, hp, inputs):
        texts = np.array(list(tfds.as_numpy(inputs))).astype(np.str)
        data = self._vectorizer.transform(texts).toarray()
        if self.selector:
            data = self.selector.transform(data).astype('float32')
        return tf.data.Dataset.from_tensor_slices(data)
