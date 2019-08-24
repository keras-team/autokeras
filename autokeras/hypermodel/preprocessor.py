import numpy as np
import tensorflow as tf
import lightgbm as lgb
from sklearn import feature_selection
from sklearn.feature_extraction import text
from tensorflow.python.util import nest

from autokeras import const
from autokeras import utils
from autokeras.hypermodel import block


class Preprocessor(block.Block):
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

    def update(self, x, y=None):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            y: EagerTensor. The targets of the tasks. Defaults to None.
        """
        raise NotImplementedError

    def transform(self, x, fit=False):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
            fit: Boolean. Whether it is in fit mode.

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

    @property
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

    def get_state(self):
        pass

    def set_state(self, state):
        pass


class Normalize(Preprocessor):
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

    def update(self, x, y=None):
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

    def transform(self, x, fit=False):
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

    @property
    def output_shape(self):
        return self.inputs[0].shape

    def get_state(self):
        return {'sum': self.sum,
                'square_sum': self.square_sum,
                'count': self.count,
                'mean': self.mean,
                'std': self.std,
                '_shape': self._shape}

    def set_state(self, state):
        self.sum = state['sum']
        self.square_sum = state['square_sum']
        self.count = state['count']
        self.mean = state['mean']
        self.std = state['std']
        self._shape = state['_shape']


class TextToIntSequence(Preprocessor):
    """Convert raw texts to sequences of word indices."""

    def __init__(self, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self._max_len = 0
        self._tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)

    def update(self, x, y=None):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        self._tokenizer.fit_on_texts([sentence])
        sequence = self._tokenizer.texts_to_sequences([sentence])[0]
        if self.max_len is None:
            self._max_len = max(self._max_len, len(sequence))

    def transform(self, x, fit=False):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        sequence = self._tokenizer.texts_to_sequences(sentence)[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            self.max_len or self._max_len)
        return sequence

    def output_types(self):
        return tf.int64,

    @property
    def output_shape(self):
        return self.max_len or self._max_len,

    def get_state(self):
        return {'max_len': self.max_len,
                '_max_len': self._max_len,
                '_tokenizer': self._tokenizer}

    def set_state(self, state):
        self.max_len = state['max_len']
        self._max_len = state['_max_len']
        self._tokenizer = state['_tokenizer']


class TextToNgramVector(Preprocessor):
    """Convert raw texts to n-gram vectors."""
    # TODO: Implement save and load.

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

    def update(self, x, y=None):
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

    def transform(self, x, fit=False):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        data = self._vectorizer.transform([sentence]).toarray()
        if self.selector:
            data = self.selector.transform(data).astype('float32')
        return data[0]

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self._shape

    def get_state(self):
        return {'_vectorizer': self._vectorizer,
                'selector': self.selector,
                'labels': self.labels,
                '_max_features': self._max_features,
                '_texts': self._texts,
                '_shape': self._shape}

    def set_state(self, state):
        self._vectorizer = state['_vectorizer']
        self.selector = state['selector']
        self.labels = state['labels']
        self._max_features = state['_max_features']
        self._vectorizer.max_features = self._max_features
        self._texts = state['_texts']
        self._shape = state['_shape']


class LgbmModule(Preprocessor):
    """Collect data, train and test the LightGBM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.label = []
        self.lgbm = lgb.LGBMModel()
        self.param = dict()
        self._one_hot_encoder = None
        self.y_shape = None

    def update(self, x, y=None):
        y = nest.flatten(y)[0].numpy()
        self.data.append(nest.flatten(x)[0].numpy())
        if not self._one_hot_encoder:
            self._one_hot_encoder = utils.OneHotEncoder()
            self._one_hot_encoder.fit_with_one_hot_encoded(np.array(y))
        self.y_shape = np.shape(y)
        y = y.reshape(1, -1)
        self.label.append(nest.flatten(self._one_hot_encoder.decode(y))[0])

    def finalize(self):
        label = np.array(self.label).flatten()
        # train_data = lgb.Dataset(np.asarray(self.data), label)
        # TODO：Split and add validation data.
        # TODO: Set hp for parameters below.
        self.param.update({'boosting_type': ['gbdt'],
                           'min_child_weight': [5],
                           'min_split_gain': [1.0],
                           'subsample': [0.8],
                           'colsample_bytree': [0.6],
                           'max_depth': [10],
                           'num_leaves': [70],
                           'learning_rate': [0.04]})
        self.param['metric'] = 'auc'
        num_round = 10
        # self.lgbm = lgb.train(self.param, train_data, num_round)
        self.lgbm.set_params(objective='regression')
        self.lgbm.fit(X=np.asarray(self.data), y=label)

    def transform(self, x, fit=False):
        ypred = [self.lgbm.predict(x.numpy().reshape((1, -1)))]
        y = self._one_hot_encoder.encode(ypred)
        y = y.reshape((-1))
        return y

    def output_types(self):
        return tf.int32,

    @property
    def output_shape(self):
        return self.y_shape

    def get_state(self):
        pass

    def set_state(self, state):
        pass
