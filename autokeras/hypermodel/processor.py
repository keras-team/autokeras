import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import re
import array
from sklearn import feature_selection
from sklearn.feature_extraction import text
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.externals.six import moves
from collections import defaultdict
from tensorflow.python.util import nest

from autokeras import const
from autokeras.hypermodel import block as hb_module


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

        # Arguments
            hp: Hyperparameters. The hyperparameters for tuning the preprocessor.
        """
        self._hp = hp

    def update(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
        """
        raise NotImplementedError

    def transform(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.

        Returns:
            A transformed instanced which can be converted to a tf.Tensor.
        """
        raise NotImplementedError

    def output_types(self):
        """The output types of the transformed data, e.g. tf.int64.

        The output types are required by tf.py_function, which is used for transform
        the dataset into a new one with a map function.

        # Returns
            A tuple of data types.
        """
        raise NotImplementedError

    def output_shape(self):
        """The output shape of the transformed data.

        The output shape is needed to build the Keras Model from the AutoModel.
        The output shape of the preprocessor is the input shape of the Keras Model.

        # Returns
            A tuple of int(s) or a TensorShape.
        """
        raise NotImplementedError

    def finalize(self):
        """Training process of the preprocessor after update with all instances."""
        pass


class OneHotEncoder(object):
    """A class that can format data.

    This class provides ways to transform data's classification label into
    vector.

    # Arguments
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

    # Arguments
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
        self.selector = None
        self.labels = None
        self._max_features = const.Constant.VOCABULARY_SIZE
        self._texts = []
        self._shape = None
        self.vocabulary = defaultdict()  # Vocabulary(Increase with the inputs)
        self.vocabulary.default_factory = self.vocabulary.__len__
        self.feature_counter = {}  # Frequency of the token in the whole doc
        self.sentence_containers = {}  # Number of sentence that contains the token
        self.tf_idf_vec = {}  # TF-IDF of all the tokens
        self.word_sum = 0  # The number of all the words in the raw doc
        self.stc_num = 0  # The number of all the sentences in the raw doc
        self.temp_vec = set()  # Store the features of each sentence, used for the transform func
        self.result = []  # Final result list of the processed text
        self.kbestfeature_value = []
        self.kbestfeature_key = []
        self.mask = []
    
    @staticmethod
    def _word_ngram(tokens, stop_words=None):
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = (1, 1)
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in moves.xrange(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in moves.xrange(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens
    
    def update(self, x):
        # TODO: Implement a sequential version fit for both
        #  TfidfVectorizer and SelectKBest
        x = nest.flatten(x)[0].numpy().decode('utf-8')
        stop_words = ENGLISH_STOP_WORDS
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = self._word_ngram(token_pattern.findall(x.lower()),  # x.lower()
                                  stop_words)
        j_indices = []
        indptr = array.array(str("i"))
        values = array.array(str("i"))
        indptr.append(0)
        set4sentence = set()
        self.stc_num += 1
        for feature in tokens:
            self.word_sum += 1
            try:
                feature_idx = self.vocabulary[feature]
                set4sentence.add(feature_idx)
                if feature_idx not in self.feature_counter:
                    self.feature_counter[feature_idx] = 1
                else:
                    self.feature_counter[feature_idx] += 1
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue
        for element in set4sentence:
            if element not in self.sentence_containers:
                self.sentence_containers[element] = 1
            else:
                self.sentence_containers[element] += 1
        set4sentence.clear()
        j_indices.extend(self.feature_counter.keys())
        values.extend(self.feature_counter.values())
        indptr.append(len(j_indices))
        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)
        vec = sp.csr_matrix((values, j_indices, indptr),
                            shape=(len(indptr) - 1, len(self.vocabulary)),
                            dtype=np.int64)
        vec.sort_indices()

    def finalize(self):
        for word in self.feature_counter:
            _tf = self.feature_counter[word] / self.word_sum
            idf = np.log(self.stc_num / self.sentence_containers[word] + 1)
            self.tf_idf_vec[word] = _tf * idf
        if len(self.vocabulary) < self._max_features:
            self._max_features = len(self.vocabulary)
        kbestfeature = dict(sorted(self.tf_idf_vec.items(),
                            key=lambda item: item[1],
                            reverse=True)[0:self._max_features])
        self.kbestfeature_value = np.array(list(dict(
            sorted(kbestfeature.items())).values()))
        self.kbestfeature_key = np.array(list(dict(
            sorted(kbestfeature.items())).keys()))
        self.mask = [0 for _ in range(self._max_features)]

    def transform(self, x):
        x = nest.flatten(x)[0].numpy().decode('utf-8')
        self._texts.append(x)
        stop_words = ENGLISH_STOP_WORDS
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = self._word_ngram(token_pattern.findall(x.lower()),
                                  stop_words)
        for feature in tokens:
            try:
                feature_idx = self.vocabulary[feature]
                self.temp_vec.add(feature_idx)
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue
        for num in self.temp_vec:
            try:
                num = list(self.kbestfeature_key).index(num)
            except ValueError:
                continue
            if num - 1 <= self._max_features:
                self.mask[num] = 1
        self.result = np.array(
            self.mask)*np.array(
            self.kbestfeature_value)
        # Refresh the mask&temp_vec for next time usage.
        self.mask = [0 for _ in range(self._max_features)]
        self.temp_vec = set()
        # TODO: For each x, what is the type of return value?
        return self.result

    def output_types(self):
        return tf.float64,

    def output_shape(self):
        return self.result.shape
