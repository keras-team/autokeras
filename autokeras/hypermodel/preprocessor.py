import ast
import re
import warnings

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow.python.util import nest

from autokeras import const
from autokeras import encoder
from autokeras import utils
from autokeras.hypermodel import base

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb


class Normalization(base.Preprocessor):
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
        self.shape = None

    def update(self, x, y=None):
        x = nest.flatten(x)[0].numpy()
        self.sum += x
        self.square_sum += np.square(x)
        self.count += 1
        self.shape = x.shape

    def finalize(self):
        axis = tuple(range(len(self.shape) - 1))
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
        return (tf.float32,)

    @property
    def output_shape(self):
        return self.shape

    def get_state(self):
        state = super().get_state()
        state.update({
            'sum': self.sum,
            'square_sum': self.square_sum,
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'shape': self.shape})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.sum = state['sum']
        self.square_sum = state['square_sum']
        self.count = state['count']
        self.mean = state['mean']
        self.std = state['std']
        self.shape = state['shape']


class TextToIntSequence(base.Preprocessor):
    """Convert raw texts to sequences of word indices."""

    def __init__(self, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.max_len_in_data = 0
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)
        self.max_len_to_use = None
        self.max_features = None

    def update(self, x, y=None):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        self.tokenizer.fit_on_texts([sentence])
        sequence = self.tokenizer.texts_to_sequences([sentence])[0]
        if self.max_len is None:
            self.max_len_in_data = max(self.max_len_in_data, len(sequence))

    def finalize(self):
        self.max_len_to_use = self.max_len or self.max_len_in_data
        self.max_features = len(self.tokenizer.word_counts) + 1

    def transform(self, x, fit=False):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        sequence = self.tokenizer.texts_to_sequences([sentence])
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            self.max_len_to_use)[0]
        return sequence

    def output_types(self):
        return (tf.int64,)

    @property
    def output_shape(self):
        return self.max_len or self.max_len_in_data,

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len})
        return config

    def get_state(self):
        state = super().get_state()
        state.update({'max_len_in_data': self.max_len_in_data,
                      'tokenizer': self.tokenizer,
                      'max_len_to_use': self.max_len_to_use,
                      'max_features': self.max_features})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.max_len_in_data = state['max_len_in_data']
        self.tokenizer = state['tokenizer']
        self.max_len_to_use = state['max_len_to_use']
        self.max_features = state['max_features']


class TextToNgramVector(base.Preprocessor):
    """Convert raw texts to n-gram vectors.

    # Arguments
        ngram_range: Int Tuple. Range of sizes of ngram tokens to be extracted.
            If not specified, it will be tuned automatically. Defaults to None.
        stop_words: Set or Iterable of strings. List of stop words to be removed
            during tokenization. Defaults to use regular expression "(?u)\b\w\w+\b".
        max_features: Positive Int. Maximum number of words to be considered during
            tokenization. Defaults to 20000.
        norm: String. Can be ('l1', 'l2' or None) Attribute to replicate
            normalization in sklearn TfidfVectorizer.
            Defaults to 'l2'.
    """

    def __init__(self,
                 ngram_range=None,
                 stop_words=None,
                 max_features=20000,
                 norm='l2',
                 **kwargs):
        super().__init__(**kwargs)
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.temp_ngram_range = None
        self.norm = norm
        if(self.norm not in ('l1', 'l2', None)):
            raise ValueError(
                "norm=%s, needs to be either 'l1','l2' or None."
                % self.norm)
        self._max_features = max_features
        if(self._max_features <= 0):
            raise ValueError(
                "max_features=%r, needs to be a positive integer."
                % self._max_features)
        self._shape = None
        self.vocabulary = {}  # Vocabulary(Increase with the inputs)
        self.sentence_containers = {}  # Number of sentence that contains the token
        self.word_count = {}  # Count of words across documents
        self._idf_vec = {}  # IDF of all the tokens
        self.stc_num = 0  # The number of all the sentences in the raw doc
        self.k_best_idf_values = []

    def build(self, hp):
        self.temp_ngram_range = self.ngram_range or (1, hp.Choice(
            'ngram_range', [1, 2], default=2))

    def _word_ngram(self, tokens):
        # handle stop words
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]

        # handle token n-grams
        min_n, max_n = self.temp_ngram_range
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

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def update(self, x, y=None):
        x = nest.flatten(x)[0].numpy().decode('utf-8')
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = self._word_ngram(token_pattern.findall(x.lower()))
        sentence_set = set()
        self.stc_num += 1
        for feature in tokens:
            if feature not in self.vocabulary:
                self.vocabulary[feature] = len(self.vocabulary)
            if(feature not in self.word_count):
                self.word_count[feature] = 1
            else:
                self.word_count[feature] += 1
            sentence_set.add(feature)
        for element in sentence_set:
            if element not in self.sentence_containers:
                self.sentence_containers[element] = 1
            else:
                self.sentence_containers[element] += 1
        sentence_set.clear()

    def finalize(self):
        self.stc_num += 1  # idf smoothing
        for word in self.sentence_containers:
            idf = (np.log((self.stc_num)/(self.sentence_containers[word] + 1)) + 1)
            self._idf_vec[word] = idf
        max_features = self._max_features
        if len(self.vocabulary) < self._max_features:
            max_features = len(self.vocabulary)
        # Sort based on term frequency to match sklearn TfidfVectorizer
        k_best_idfs = dict(sorted(self._idf_vec.items(),
                                  key=(lambda item: self.word_count[item[0]]),
                                  reverse=True)[0:max_features])
        self.k_best_idf_values = np.array(list(dict(
            sorted(k_best_idfs.items())).values()))
        k_best_idf_keys = np.array(list(dict(
            sorted(k_best_idfs.items())).keys()))
        self.vocabulary.clear()
        for i in range(max_features):
            self.vocabulary[k_best_idf_keys[i]] = i
        self._shape = np.shape(self.k_best_idf_values)

    def transform(self, x, fit=False):
        # Calculate tf at doc level
        tf = np.zeros(len(self.vocabulary), dtype=int)
        x = nest.flatten(x)[0].numpy().decode('utf-8')
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = self._word_ngram(token_pattern.findall(x.lower()))

        for feature in tokens:
            if feature in self.vocabulary:
                feature_idx = self.vocabulary[feature]
                tf[feature_idx] += 1
        result = tf * self.k_best_idf_values
        result = normalize([result], norm=self.norm, copy=False)[0]
        return result

    def output_types(self):
        return (tf.float64,)

    @property
    def output_shape(self):
        return self._shape

    def get_config(self):
        config = super().get_config()
        config.update({'ngram_range': self.ngram_range,
                       'max_features': self._max_features,
                       'stop_words': self.stop_words,
                       'norm': self.norm})
        return config

    def get_state(self):
        state = super().get_state()
        state.update({'shape': self._shape,
                      'k_best_idf_values': self.k_best_idf_values,
                      'vocabulary': self.vocabulary,
                      'idf_vec': self._idf_vec,
                      'sentence_containers': self.sentence_containers,
                      'word_count': self.word_count,
                      'stc_num': self.stc_num})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._shape = state['shape']
        self.k_best_idf_values = state['k_best_idf_values']
        self.vocabulary = state['vocabulary']
        self._idf_vec = state['idf_vec']
        self.sentence_containers = state['sentence_containers']
        self.word_count = state['word_count']
        self.stc_num = state['stc_num']


class LightGBMModel(base.Preprocessor):
    """The base class for LightGBMClassifier and LightGBMRegressor.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.targets = []
        self._output_shape = None
        self.lgbm = None
        self.seed = seed
        self.params = None

    def update(self, x, y=None):
        """ Store the train data and decode.

        # Arguments
            x: Eager Tensor. The data to be stored.
            y: Eager Tensor. The label to be stored.
        """
        self.data.append(nest.flatten(x)[0].numpy())
        self.targets.append(nest.flatten(y)[0].numpy())

    def transform(self, x, fit=False):
        """ Transform the data using well-trained LightGBM regressor.

        # Arguments
            x: Eager Tensor. The data to be transformed.

        # Returns
            Eager Tensor. The predicted value of x.
        """
        return [self.lgbm.predict(x.numpy().reshape((1, -1)))]

    def build(self, hp):
        self.params = {
            'boosting_type': ['gbdt'],
            'min_child_weight': hp.Choice('min_child_weight',
                                          [5, 10, 30, 50, 60, 80, 100],
                                          default=5),
            'min_split_gain': [1.0],
            'subsample': [0.8],
            'colsample_bytree': hp.Choice('colsample_bytree',
                                          [0.6, 0.7],
                                          default=0.6),
            'max_depth': hp.Choice('max_depth',
                                   [5, 8, 10],
                                   default=10),
            'num_leaves': [70],
            'learning_rate': hp.Choice('learning_rate',
                                       [0.03, 0.045, 0.06, 0.075,
                                        0.85, 0.95, 0.105, 0.12],
                                       default=0.105),
            'n_estimators': hp.Choice('n_estimators',
                                      [50, 100, 150, 200],
                                      default=50)}

    def finalize(self):
        """ Train the LightGBM model with the data and value stored."""
        target = np.array(self.targets).flatten()
        # TODO: Set hp for parameters below.
        self.lgbm.set_params(**self.params)
        self.lgbm.fit(X=np.asarray(self.data), y=target)
        self.data = []
        self.targets = []

    @property
    def output_shape(self):
        return self._output_shape

    def output_types(self):
        return (tf.float32,)

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

    def set_state(self, state):
        super().set_state(state)
        self.lgbm = state['lgbm']
        self._output_shape = state['output_shape']
        self.params = state['params']

    def get_state(self):
        state = super().get_state()
        state.update({'lgbm': self.lgbm,
                      'output_shape': self._output_shape,
                      'params': self.params})
        return state


class LightGBMClassifier(LightGBMModel):
    """Collect data, train and test the LightGBM for classification task.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.lgbm = lgb.LGBMClassifier(random_state=self.seed)
        self._one_hot_encoder = None
        self.num_classes = None

    def finalize(self):
        self._output_shape = self.targets[0].shape
        if self.num_classes > 2:
            self._one_hot_encoder = encoder.OneHotEncoder()
            self._one_hot_encoder.fit_with_one_hot_encoded(self.targets)
            self.targets = self._one_hot_encoder.decode(self.targets)
        super().finalize()

    def get_params(self):
        params = super().get_params()
        params['eval_metric'] = 'logloss'
        return params

    def transform(self, x, fit=False):
        """ Transform the data using well-trained LightGBM classifier.

        # Arguments
            x: Eager Tensor. The data to be transformed.

        # Returns
            Eager Tensor. The predicted label of x.
        """
        y = super().transform(x, fit)
        if self._one_hot_encoder:
            y = self._one_hot_encoder.encode(y)
            y = y.reshape((-1))
        return y

    def set_state(self, state):
        super().set_state(state)
        self._one_hot_encoder = encoder.deserialize(state['one_hot_encoder'])
        if self._one_hot_encoder:
            self._one_hot_encoder.set_state(state['one_hot_encoder_state'])
        self.num_classes = state['num_classes']

    def get_state(self):
        state = super().get_state()
        state.update({'one_hot_encoder': encoder.serialize(self._one_hot_encoder),
                      'one_hot_encoder_state': self._one_hot_encoder.get_state() if
                      self._one_hot_encoder else None,
                      'num_classes': self.num_classes})
        return state


class LightGBMRegressor(LightGBMModel):
    """Collect data, train and test the LightGBM for regression task.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.lgbm = lgb.LGBMRegressor(random_state=self.seed)

    def finalize(self):
        self._output_shape = self.targets[0].shape
        super().finalize()


class LightGBM(base.Preprocessor):
    """LightGBM Block for classification or regression task.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.lightgbm_block = None
        self.seed = seed

    def build(self, hp):
        self.lightgbm_block.build(hp)

    def get_state(self):
        return {
            'lightgbm_block': tf.keras.utils.serialize_keras_object(
                self.lightgbm_block),
            'lightgbm_block_state': self.lightgbm_block.get_state()
            if self.lightgbm_block else None}

    def set_state(self, state):
        self.lightgbm_block = tf.keras.utils.deserialize_keras_object(
            state['lightgbm_block'],
            module_objects=globals())
        if self.lightgbm_block:
            self.lightgbm_block.set_state(state['lightgbm_block_state'])

    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

    def update(self, x, y=None):
        self.lightgbm_block.update(x, y)

    def transform(self, x, fit=False):
        return self.lightgbm_block.transform(x, fit)

    def finalize(self):
        self.lightgbm_block.finalize()

    def output_types(self):
        return self.lightgbm_block.output_types()

    @property
    def output_shape(self):
        return self.lightgbm_block.output_shape


class ImageAugmentation(base.Preprocessor):
    """Collection of various image augmentation methods.

    # Arguments
        percentage: Float. The percentage of data to augment.
        rotation_range: Int. The value can only be 0, 90, or 180.
            Degree range for random rotations. Default to 180.
        random_crop: Boolean. Whether to crop the image randomly. Default to True.
        brightness_range: Positive float.
            Serve as 'max_delta' in tf.image.random_brightness. Default to 0.5.
            Equivalent to adjust brightness using a 'delta' randomly picked in
            the interval [-max_delta, max_delta).
        saturation_range: Positive float or Tuple.
            If given a positive float, _get_min_and_max() will automated generate
            a tuple for saturation range. If given a tuple directly, it will serve
            as a range for picking a saturation shift value from. Default to 0.5.
        contrast_range: Positive float or Tuple.
            If given a positive float, _get_min_and_max() will automated generate
            a tuple for contrast range. If given a tuple directly, it will serve
            as a range for picking a contrast shift value from. Default to 0.5.
        translation: Boolean. Whether to translate the image.
        horizontal_flip: Boolean. Whether to flip the image horizontally.
        vertical_flip: Boolean. Whether to flip the image vertically.
        gaussian_noise: Boolean. Whether to add gaussian noise to the image.
    """

    def __init__(self,
                 percentage=0.25,
                 rotation_range=180,
                 random_crop=True,
                 brightness_range=0.5,
                 saturation_range=0.5,
                 contrast_range=0.5,
                 translation=True,
                 horizontal_flip=True,
                 vertical_flip=True,
                 gaussian_noise=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.percentage = percentage
        self.rotation_range = rotation_range
        self._rotate_choices = [0]
        if self.rotation_range == 90:
            self._rotate_choices = [0, 1, 3]
        elif self.rotation_range == 180:
            self._rotate_choices = [0, 1, 2, 3]
        self.random_crop = random_crop
        if self.random_crop:
            # Generate 20 crop settings, ranging from a 1% to 20% crop.
            self.scales = list(np.arange(0.8, 1.0, 0.01))
            self.boxes = np.zeros((len(self.scales), 4))
            for i, scale in enumerate(self.scales):
                x1 = y1 = 0.5 - (0.5 * scale)
                x2 = y2 = 0.5 + (0.5 * scale)
                self.boxes[i] = [x1, y1, x2, y2]
        self.brightness_range = brightness_range
        self.saturation_range = self._get_min_and_max(saturation_range,
                                                      'saturation_range')
        self.contrast_range = self._get_min_and_max(contrast_range,
                                                    'contrast_range')
        self.translation = translation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.gaussian_noise = gaussian_noise
        self.shape = None

    @staticmethod
    def _get_min_and_max(value, name):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            min_value, max_value = value
            return min_value, max_value
        elif isinstance(value, (int, float)):
            min_value = 1. - value
            max_value = 1. + value
            return min_value, max_value
        elif value == 0:
            return None
        else:
            raise ValueError('Expected {name} to be either a float between 0 and 1, '
                             'or a tuple of 2 floats between 0 and 1, '
                             'but got {value}'.format(name=name, value=value))

    def update(self, x, y=None):
        x = nest.flatten(x)[0].numpy()
        self.shape = x.shape

    def transform(self, x, fit=False):
        self.shape = x.shape
        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if fit and choice < self.percentage:
            return self._augment(x)
        return x

    def _rotate(self, x):
        rotate_choice = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=len(self._rotate_choices),
            dtype=tf.int64)
        return tf.image.rot90(x, k=self._rotate_choices[rotate_choice])

    def _random_crop(self, x):
        crops = tf.image.crop_and_resize(
            [x],
            boxes=self.boxes,
            box_indices=np.zeros(len(self.scales)),
            crop_size=self.shape[:2])
        return crops[tf.random.uniform(shape=[],
                                       minval=0,
                                       maxval=len(self.scales),
                                       dtype=tf.int32)]

    def _augment(self, x):
        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.rotation_range != 0 and choice < 0.5:
            x = self._rotate(x)

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.random_crop and choice < 0.5:
            x = self._random_crop(x)

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.brightness_range != 0 and choice < 0.5:
            x = tf.image.random_brightness(x,
                                           self.brightness_range)

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.saturation_range and self.shape[-1] == 3 and choice > 0.5:
            x = tf.image.random_saturation(x,
                                           self.saturation_range[0],
                                           self.saturation_range[1])

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if self.contrast_range and choice < 0.5:
            x = tf.image.random_contrast(x,
                                         self.contrast_range[0],
                                         self.contrast_range[1])

        if self.horizontal_flip:
            x = tf.image.random_flip_left_right(x)

        if self.vertical_flip:
            x = tf.image.random_flip_up_down(x)

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        noise = tf.random.normal(shape=tf.shape(x),
                                 mean=0.0,
                                 stddev=1.0,
                                 dtype=tf.float32)
        if self.gaussian_noise and choice < 0.5:
            x = tf.add(x, noise)
        return x

    def output_types(self):
        return (tf.float32,)

    @property
    def output_shape(self):
        return self.shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'percentage': self.percentage,
            'rotation_range': self.rotation_range,
            'random_crop': self.random_crop,
            'brightness_range': self.brightness_range,
            'saturation_range': self.saturation_range,
            'contrast_range': self.contrast_range,
            'translation': self.translation,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'gaussian_noise': self.gaussian_noise})
        return config

    def get_state(self):
        state = super().get_state()
        state.update({'shape': self.shape})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.shape = state['shape']


class FeatureEngineering(base.Preprocessor):
    """A preprocessor block does feature engineering for the data.

    # Arguments
        max_columns: Int. The maximum number of columns after feature engineering.
            Defaults to 1000.
    """

    def __init__(self, max_columns=1000, **kwargs):
        super().__init__(**kwargs)
        self.column_names = None
        self.column_types = None
        self.max_columns = max_columns
        self.num_columns = 0
        self.num_rows = 0
        self.shape = None
        # A list of categorical column indices.
        self.categorical_col = []
        # A list of numerical column indices.
        self.numerical_col = []
        self.label_encoders = {}
        self.value_counters = {}
        self.categorical_categorical = {}
        self.numerical_categorical = {}
        self.count_frequency = {}
        # more than 32, less than 100
        self.high_level1_col = []
        # more than 100
        self.high_level2_col = []
        self.high_level_cat_cat = {}
        self.high_level_num_cat = {}

    def initialize(self):
        for column_name, column_type in self.column_types.items():
            if column_type == 'categorical':
                self.categorical_col.append(
                    self.column_names.index(column_name))
            elif column_type == 'numerical':
                self.numerical_col.append(
                    self.column_names.index(column_name))
            else:
                raise ValueError('Unsupported column type: '
                                 '{type}'.format(type=column_type))

        for index, cat_col_index1 in enumerate(self.categorical_col):
            self.label_encoders[cat_col_index1] = encoder.LabelEncoder()
            self.value_counters[cat_col_index1] = {}
            self.count_frequency[cat_col_index1] = {}
            for cat_col_index2 in self.categorical_col[index + 1:]:
                self.categorical_categorical[(cat_col_index1, cat_col_index2)] = {}
            for num_col_index in self.numerical_col:
                self.numerical_categorical[(num_col_index, cat_col_index1)] = {}

    def update(self, x, y=None):
        if self.num_rows == 0:
            self.num_columns = len(self.column_types)
            self.initialize()

        self.num_rows += 1
        x = nest.flatten(x)[0].numpy()

        self.fill_missing(x)

        for col_index in self.categorical_col:
            key = str(x[col_index])
            self.label_encoders[col_index].update(key)
            self.value_counters[col_index].setdefault(key, 0)
            self.value_counters[col_index][key] += 1

        for col_index1, col_index2 in self.categorical_categorical.keys():
            key = str((x[col_index1], x[col_index2]))
            self.categorical_categorical[(col_index1, col_index2)].setdefault(key, 0)
            self.categorical_categorical[(col_index1, col_index2)][key] += 1

        for num_col_index, cat_col_index in self.numerical_categorical.keys():
            key = str(x[cat_col_index])
            v = x[num_col_index]
            self.numerical_categorical[(
                num_col_index, cat_col_index)].setdefault(key, 0)
            self.numerical_categorical[(num_col_index, cat_col_index)][key] += v

    def transform(self, x, fit=False):
        x = nest.flatten(x)[0].numpy()

        self.fill_missing(x)

        new_values = []
        # append frequency
        for col_index in self.high_level1_col:
            cat_name = str(x[col_index])
            new_value = self.count_frequency[col_index][cat_name] if \
                cat_name in self.count_frequency[col_index] else -1
            new_values.append(new_value)

        # append cat-cat value
        for key, value in self.high_level_cat_cat.items():
            col_index1, col_index2 = key
            pair = str((x[col_index1], x[col_index2]))
            new_value = value[pair] if pair in value else -1
            new_values.append(new_value)

        # append num-cat value
        for key, value in self.high_level_num_cat.items():
            num_col_index, cat_col_index = key
            cat_name = str(x[cat_col_index])
            new_value = value[cat_name] if cat_name in value else -1
            new_values.append(new_value)

        # LabelEncoding
        for col_index in self.categorical_col:
            key = str(x[col_index])
            try:
                x[col_index] = self.label_encoders[col_index].transform(key)
            except KeyError:
                x[col_index] = -1
        return np.hstack((x, np.array(new_values)))

    def fill_missing(self, x):
        for col_index in range(self.num_columns):
            x[col_index] = x[col_index].decode('utf-8')
            if col_index in self.numerical_col:
                if x[col_index] == 'nan':
                    x[col_index] = 0.0
                else:
                    x[col_index] = float(x[col_index])
            else:
                if x[col_index] == 'nan':
                    x[col_index] = 0

    def finalize(self):
        # divide column according to category number of the column
        for col_index in self.value_counters.keys():
            num_cat = len(self.value_counters[col_index])
            if num_cat > 32 and num_cat <= 100:
                self.high_level1_col.append(col_index)
                self.count_frequency[col_index] = {
                    key: value / (self.num_rows * num_cat)
                    for key, value in self.value_counters[col_index].items()}
            elif num_cat > 100:
                self.high_level2_col.append(col_index)

        self.high_level2_col.sort()

        for index, cat_col_index1 in enumerate(self.high_level2_col):
            # extract high level columns from cat-cat dict
            for cat_col_index2 in self.high_level2_col[index + 1:]:
                pair = (cat_col_index1, cat_col_index2)
                self.high_level_cat_cat[pair] = self.categorical_categorical[pair]

            # extract high level columns from num-cat dict and calculate mean
            for num_col_index in self.numerical_col:
                pair = (num_col_index, cat_col_index1)
                self.high_level_num_cat[pair] = self.numerical_categorical[pair]
                for key, value in self.high_level_num_cat[pair].items():
                    self.high_level_num_cat[pair][key] /= self.value_counters[
                        cat_col_index1][key]

        self.shape = (len(self.column_types)
                      + len(self.high_level1_col)
                      + len(self.high_level_cat_cat)
                      + len(self.high_level_num_cat),)

    def output_types(self):
        return (tf.float32,)

    @property
    def output_shape(self):
        return self.shape

    def get_state(self):
        state = super().get_state()
        label_encoders = {
            key: encoder.serialize(label_encoder)
            for key, label_encoder in self.label_encoders.items()}
        label_encoders_state = {
            key: label_encoder.get_state()
            for key, label_encoder in self.label_encoders.items()}
        state.update({
            'shape': self.shape,
            'num_rows': self.num_rows,
            'categorical_col': self.categorical_col,
            'numerical_col': self.numerical_col,
            'label_encoders': utils.to_type_key(label_encoders, str),
            'label_encoders_state': utils.to_type_key(label_encoders_state, str),
            'value_counters': utils.to_type_key(self.value_counters, str),
            'categorical_categorical': utils.to_type_key(
                self.categorical_categorical, str),
            'numerical_categorical': utils.to_type_key(
                self.numerical_categorical, str),
            'count_frequency': utils.to_type_key(self.count_frequency, str),
            'high_level1_col': self.high_level1_col,
            'high_level2_col': self.high_level2_col,
            'high_level_cat_cat': utils.to_type_key(self.high_level_cat_cat, str),
            'high_level_num_cat': utils.to_type_key(self.high_level_num_cat, str),
            'column_names': self.column_names,
            'column_types': utils.to_type_key(self.column_types, str),
            'num_columns': self.num_columns,
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        for key, label_encoder in state['label_encoders'].items():
            self.label_encoders[key] = encoder.deserialize(label_encoder)
        for key, label_encoder_state in state['label_encoders_state'].items():
            self.label_encoders[key].set_state(label_encoder_state)
        self.column_names = state['column_names']
        self.column_types = state['column_types']
        self.num_columns = state['num_columns']
        self.shape = state['shape']
        self.num_rows = state['num_rows']
        self.categorical_col = state['categorical_col']
        self.numerical_col = state['numerical_col']
        self.value_counters = utils.to_type_key(state['value_counters'], int)
        self.categorical_categorical = utils.to_type_key(
            state['categorical_categorical'], ast.literal_eval)
        self.numerical_categorical = utils.to_type_key(
            state['numerical_categorical'], ast.literal_eval)
        self.count_frequency = utils.to_type_key(state['count_frequency'], int)
        self.high_level1_col = state['high_level1_col']
        self.high_level2_col = state['high_level2_col']
        self.high_level_cat_cat = utils.to_type_key(
            state['high_level_cat_cat'], ast.literal_eval)
        self.high_level_num_cat = utils.to_type_key(
            state['high_level_num_cat'], ast.literal_eval)

    def get_config(self):
        config = super().get_config()
        config.update({'max_columns': self.max_columns})
        return config
