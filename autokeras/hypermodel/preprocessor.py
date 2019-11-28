import ast
import random
import warnings

import numpy as np
import tensorflow as tf
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from collections import defaultdict
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

    def get_weights(self):
        return {'sum': self.sum,
                'square_sum': self.square_sum,
                'count': self.count,
                'mean': self.mean,
                'std': self.std,
                'shape': self.shape}

    def set_weights(self, weights):
        self.sum = weights['sum']
        self.square_sum = weights['square_sum']
        self.count = weights['count']
        self.mean = weights['mean']
        self.std = weights['std']
        self.shape = weights['shape']


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
        return {'max_len': self.max_len}

    def set_config(self, config):
        self.max_len = config['max_len']

    def get_weights(self):
        return {'max_len_in_data': self.max_len_in_data,
                'tokenizer': self.tokenizer,
                'max_len_to_use': self.max_len_to_use,
                'max_features': self.max_features}

    def set_weights(self, weights):
        self.max_len_in_data = weights['max_len_in_data']
        self.tokenizer = weights['tokenizer']
        self.max_len_to_use = weights['max_len_to_use']
        self.max_features = weights['max_features']


class TextToNgramVector(base.Preprocessor):
    """Convert raw texts to n-gram vectors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selector = None
        self.targets = None
        self._max_features = const.Constant.VOCABULARY_SIZE
        self._shape = None
        self.vocabulary = defaultdict()  # Vocabulary(Increase with the inputs)
        self.vocabulary.default_factory = self.vocabulary.__len__
        self.feature_counter = {}  # Frequency of the token in the whole doc
        self.sentence_containers = {}  # Number of sentence that contains the token
        self.tf_idf_vec = {}  # TF-IDF of all the tokens
        self.word_sum = 0  # The number of all the words in the raw doc
        self.stc_num = 0  # The number of all the sentences in the raw doc
        self.temp_vec = set()  # Store the features of each sentencey
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

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def update(self, x, y=None):
        # TODO: Implement a sequential version fit for both
        #  TfidfVectorizer and SelectKBest
        x = nest.flatten(x)[0].numpy().decode('utf-8')
        stop_words = ENGLISH_STOP_WORDS
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = self._word_ngram(token_pattern.findall(x.lower()),
                                  stop_words)
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
        self.mask = np.zeros(self._max_features, dtype=int)
        self._shape = np.shape(self.mask)

    def transform(self, x, fit=False):
        x = nest.flatten(x)[0].numpy().decode('utf-8')
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

        self.result = self.mask * self.kbestfeature_value
        # Refresh the mask&temp_vec for next time usage.
        self.mask = np.zeros(self._max_features, dtype=int)
        self.temp_vec = set()
        # TODO: For each x, what is the type of return value?
        return self.result

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self._shape

    def get_weights(self):
        return {'selector': self.selector,
                'targets': self.targets,
                'max_features': self._max_features,
                'shape': self._shape,
                'kbestfeature_value': self.kbestfeature_value,
                'kbestfeature_key': self.kbestfeature_key}

    def set_weights(self, weights):
        self.selector = weights['selector']
        self.targets = weights['targets']
        self._max_features = weights['max_features']
        self._shape = weights['shape']
        self.kbestfeature_value = weights['kbestfeature_value']
        self.kbestfeature_key = weights['kbestfeature_key']
        self.mask = np.zeros(self._max_features, dtype=int)


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

    def set_weights(self, weights):
        self.lgbm = weights['lgbm']
        self._output_shape = weights['output_shape']
        self.seed = weights['seed']
        self.params = weights['params']

    def get_weights(self):
        return {'lgbm': self.lgbm,
                'output_shape': self._output_shape,
                'seed': self.seed,
                'params': self.params}


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

    def set_weights(self, weights):
        super().set_weights(weights)
        self._one_hot_encoder = encoder.OneHotEncoder()
        self._one_hot_encoder.set_state(weights['one_hot_encoder'])
        self.num_classes = weights['num_classes']

    def get_weights(self):
        weights = super().get_weights()
        weights.update({'one_hot_encoder': self._one_hot_encoder.get_state(),
                        'num_classes': self.num_classes})
        return weights


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


class LightGBMBlock(base.Preprocessor):
    """LightGBM Block for classification or regression task.

    # Arguments
        seed: Int. Random seed.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.lightgbm_block = None
        self.heads = None
        self.seed = seed

    def build(self, hp):
        self.lightgbm_block.build(hp)

    def get_weights(self):
        return self.lightgbm_block.get_weights()

    def set_weights(self, weights):
        self.lightgbm_block.set_weights(weights)

    def get_config(self):
        return self.lightgbm_block.get_config()

    def set_config(self, config):
        self.lightgbm_block.set_config(config)

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

    def set_hp(self, hp):
        self.lightgbm_block.set_hp(hp)


class ImageAugmentation(base.Preprocessor):
    """Collection of various image augmentation methods.

    # Arguments
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
        seed: Int. Seed for tf.image.random_*(). Default to None.
    """

    def __init__(self,
                 rotation_range=180,
                 random_crop=True,
                 brightness_range=0.5,
                 saturation_range=0.5,
                 contrast_range=0.5,
                 translation=True,
                 horizontal_flip=True,
                 vertical_flip=True,
                 gaussian_noise=True,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rotation_range = rotation_range
        self.random_crop = random_crop
        self.brightness_range = brightness_range
        self.saturation_range = self._get_min_and_max(saturation_range,
                                                      'saturation_range')
        self.contrast_range = self._get_min_and_max(contrast_range,
                                                    'contrast_range')
        self.translation = translation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.gaussian_noise = gaussian_noise
        self.seed = seed
        self.shape = None

    @staticmethod
    def _get_min_and_max(value, name):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    'Argument %s expected either a float between 0 and 1, '
                    'or a tuple of 2 floats between 0 and 1, '
                    'but got: %s' % (value, name))
            min_value, max_value = value
        elif value == 0:
            return None
        min_value = 1. - value
        max_value = 1. + value
        return min_value, max_value

    def update(self, x, y=None):
        x = nest.flatten(x)[0].numpy()
        self.shape = x.shape

    def transform(self, x, fit=False):
        if not fit:
            return x
        np.random.seed(self.seed)
        self.shape = x.shape
        target_height, target_width, channels = self.shape
        k_choices = {}
        if self.rotation_range == 0:
            k_choices = [0]
        elif self.rotation_range == 90:
            k_choices = [0, 1, 3]
        elif self.rotation_range == 180:
            k_choices = [0, 1, 2, 3]
        x = tf.image.rot90(x, k=random.choice(k_choices))

        if self.random_crop:
            crop_height = np.random.randint(low=1, high=target_height)
            crop_width = np.random.randint(low=1, high=target_width)
            crop_size = [crop_height,
                         crop_width,
                         channels]
            target_shape = (target_height, target_width)
            x = tf.image.resize(
                tf.image.random_crop(x, size=crop_size, seed=self.seed),
                size=target_shape)

        if self.brightness_range != 0:
            x = tf.image.random_brightness(x, self.brightness_range, self.seed)

        if self.saturation_range is not None and channels == 3:
            min_value, max_value = self.saturation_range
            x = tf.image.random_saturation(x, min_value, max_value, self.seed)

        if self.contrast_range is not None:
            min_value, max_value = self.contrast_range
            x = tf.image.random_contrast(x, min_value, max_value, self.seed)

        if self.translation:
            pad_top = np.random.randint(low=0,
                                        high=max(int(target_height*0.3), 1))
            pad_left = np.random.randint(low=0,
                                         high=max(int(target_width*0.3), 1))
            pad_bottom = np.random.randint(low=0,
                                           high=max(int(target_height*0.3), 1))
            pad_right = np.random.randint(low=0,
                                          high=max(int(target_width*0.3), 1))
            x = tf.image.pad_to_bounding_box(x, pad_top, pad_left,
                                             target_height + pad_bottom + pad_top,
                                             target_width + pad_right + pad_left)
            x = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right,
                                              target_height, target_width)

        if self.horizontal_flip:
            x = tf.image.flip_left_right(x)

        if self.vertical_flip:
            x = tf.image.flip_up_down(x)

        if self.gaussian_noise:
            noise = tf.random.normal(shape=tf.shape(x),
                                     mean=0.0,
                                     stddev=1.0,
                                     seed=self.seed,
                                     dtype=tf.float32)
            x = tf.add(x, noise)
        return x

    def output_types(self):
        return (tf.float32,)

    @property
    def output_shape(self):
        return self.shape

    def get_config(self):
        return {'rotation_range': self.rotation_range,
                'random_crop': self.random_crop,
                'brightness_range': self.brightness_range,
                'saturation_range': self.saturation_range,
                'contrast_range': self.contrast_range,
                'translation': self.translation,
                'horizontal_flip': self.horizontal_flip,
                'vertical_flip': self.vertical_flip,
                'gaussian_noise': self.gaussian_noise,
                'seed': self.seed,
                'shape': self.shape}

    def set_config(self, config):
        self.rotation_range = config['rotation_range']
        self.random_crop = config['random_crop']
        self.brightness_range = config['brightness_range']
        self.saturation_range = config['saturation_range']
        self.contrast_range = config['contrast_range']
        self.translation = config['translation']
        self.horizontal_flip = config['horizontal_flip']
        self.vertical_flip = config['vertical_flip']
        self.gaussian_noise = config['gaussian_noise']
        self.seed = config['seed']
        self.shape = config['shape']


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

    def get_weights(self):
        label_encoders_state = {
            key: label_encoder.get_state()
            for key, label_encoder in self.label_encoders.items()}
        return {
            'shape': self.shape,
            'num_rows': self.num_rows,
            'categorical_col': self.categorical_col,
            'numerical_col': self.numerical_col,
            'label_encoders': utils.to_type_key(label_encoders_state, str),
            'value_counters': utils.to_type_key(self.value_counters, str),
            'categorical_categorical': utils.to_type_key(
                self.categorical_categorical, str),
            'numerical_categorical': utils.to_type_key(
                self.numerical_categorical, str),
            'count_frequency': utils.to_type_key(self.count_frequency, str),
            'high_level1_col': self.high_level1_col,
            'high_level2_col': self.high_level2_col,
            'high_level_cat_cat': utils.to_type_key(self.high_level_cat_cat, str),
            'high_level_num_cat': utils.to_type_key(self.high_level_num_cat, str)}

    def set_weights(self, weights):
        for key, label_encoder_state in utils.to_type_key(weights['label_encoders'],
                                                          int).items():
            self.label_encoders[key] = encoder.LabelEncoder()
            self.label_encoders[key].set_state(label_encoder_state)
        self.shape = weights['shape']
        self.num_rows = weights['num_rows']
        self.categorical_col = weights['categorical_col']
        self.numerical_col = weights['numerical_col']
        self.value_counters = utils.to_type_key(weights['value_counters'], int)
        self.categorical_categorical = utils.to_type_key(
            weights['categorical_categorical'], ast.literal_eval)
        self.numerical_categorical = utils.to_type_key(
            weights['numerical_categorical'], ast.literal_eval)
        self.count_frequency = utils.to_type_key(weights['count_frequency'], int)
        self.high_level1_col = weights['high_level1_col']
        self.high_level2_col = weights['high_level2_col']
        self.high_level_cat_cat = utils.to_type_key(
            weights['high_level_cat_cat'], ast.literal_eval)
        self.high_level_num_cat = utils.to_type_key(
            weights['high_level_num_cat'], ast.literal_eval)

    def get_config(self):
        return {'column_names': self.column_names,
                'column_types': utils.to_type_key(self.column_types, str),
                'num_columns': self.num_columns,
                'max_columns': self.max_columns}

    def set_config(self, config):
        self.column_names = config['column_names']
        self.column_types = config['column_types']
        self.num_columns = config['num_columns']
        self.max_columns = config['max_columns']
