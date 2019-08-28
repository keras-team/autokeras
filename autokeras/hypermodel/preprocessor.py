import collections
import random

import numpy as np
import warnings
import tensorflow as tf
from sklearn import feature_selection
from sklearn.feature_extraction import text
from tensorflow.python.util import nest

from autokeras import const
from autokeras import utils
from autokeras.hypermodel import block

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb


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

    def get_config(self):
        """Get the configuration of the preprocessor.

        # Returns
            A dictionary of configurations of the preprocessor.
        """
        return {}

    def set_config(self, config):
        """Set the configuration of the preprocessor.

        # Arguments
            config: A dictionary of the configurations of the preprocessor.
        """
        pass

    def clear_weights(self):
        """Delete the trained weights of the preprocessor."""
        pass

    def get_weights(self):
        """Get the trained weights of the preprocessor.

        # Returns
            A dictionary of trained weights of the preprocessor.
        """
        return {}

    def set_weights(self, weights):
        """Set the trained weights of the preprocessor.

        # Arguments
            weights: A dictionary of trained weights of the preprocessor.
        """
        pass


class Normalization(Preprocessor):
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
        return tf.float64,

    @property
    def output_shape(self):
        return self.inputs[0].shape

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

    def clear_weights(self):
        self.sum = 0
        self.square_sum = 0
        self.count = 0
        self.mean = None
        self.std = None
        self.shape = None


class TextToIntSequence(Preprocessor):
    """Convert raw texts to sequences of word indices."""

    def __init__(self, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.max_len_in_data = 0
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)

    def update(self, x, y=None):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        self.tokenizer.fit_on_texts([sentence])
        sequence = self.tokenizer.texts_to_sequences([sentence])[0]
        if self.max_len is None:
            self.max_len_in_data = max(self.max_len_in_data, len(sequence))

    def transform(self, x, fit=False):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        sequence = self.tokenizer.texts_to_sequences(sentence)[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            self.max_len or self.max_len_in_data)
        return sequence

    def output_types(self):
        return tf.int64,

    @property
    def output_shape(self):
        return self.max_len or self.max_len_in_data,

    def get_config(self):
        return {'max_len': self.max_len}

    def set_config(self, config):
        self.max_len = config['max_len']

    def get_weights(self):
        return {'max_len_in_data': self.max_len_in_data,
                'tokenizer': self.tokenizer}

    def set_weights(self, weights):
        self.max_len_in_data = weights['max_len_in_data']
        self.tokenizer = weights['tokenizer']

    def clear_weights(self):
        self.max_len_in_data = 0
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=const.Constant.VOCABULARY_SIZE)


class TextToNgramVector(Preprocessor):
    """Convert raw texts to n-gram vectors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = text.TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',
            min_df=2)
        self.selector = None
        self.targets = None
        self.vectorizer.max_features = const.Constant.VOCABULARY_SIZE
        self._texts = []
        self.shape = None

    def update(self, x, y=None):
        # TODO: Implement a sequential version fit for both
        #  TfidfVectorizer and SelectKBest
        self._texts.append(nest.flatten(x)[0].numpy().decode('utf-8'))

    def finalize(self):
        self._texts = np.array(self._texts)
        self.vectorizer.fit(self._texts)
        data = self.vectorizer.transform(self._texts)
        self.shape = data.shape[1:]
        if self.targets:
            self.selector = feature_selection.SelectKBest(
                feature_selection.f_classif,
                k=min(self.vectorizer.max_features, data.shape[1]))
            self.selector.fit(data, self.targets)

    def transform(self, x, fit=False):
        sentence = nest.flatten(x)[0].numpy().decode('utf-8')
        data = self.vectorizer.transform([sentence]).toarray()
        if self.selector:
            data = self.selector.transform(data).astype('float32')
        return data[0]

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self.shape

    def get_weights(self):
        return {'vectorizer': self.vectorizer,
                'selector': self.selector,
                'targets': self.targets,
                'max_features': self.vectorizer.max_features,
                'texts': self._texts,
                'shape': self.shape}

    def set_weights(self, weights):
        self.vectorizer = weights['vectorizer']
        self.selector = weights['selector']
        self.targets = weights['targets']
        self.vectorizer.max_features = weights['max_features']
        self._texts = weights['texts']
        self.shape = weights['shape']

    def clear_weights(self):
        self.vectorizer = text.TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',
            min_df=2)
        self.selector = None
        self.targets = None
        self.vectorizer.max_features = const.Constant.VOCABULARY_SIZE
        self._texts = []
        self.shape = None


class LightGBMModel(Preprocessor):
    """The base class for LightGBMClassifier and LightGBMRegressor."""

    def update(self, x, y=None):
        raise NotImplementedError

    def output_types(self):
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.targets = []
        self.y_shape = None
        self.lgbm = None

    def transform(self, x, fit=False):
        """ Transform the data using well-trained LightGBM regressor.

        # Arguments
            x: Eager Tensor. The data to be transformed.

        # Returns
            Eager Tensor. The predicted value of x.
        """
        return [self.lgbm.predict(x.numpy().reshape((1, -1)))]

    def get_params(self):
        return {'boosting_type': ['gbdt'],
                'min_child_weight': [5],
                'min_split_gain': [1.0],
                'subsample': [0.8],
                'colsample_bytree': [0.6],
                'max_depth': [10],
                'num_leaves': [70],
                'learning_rate': [0.04]}

    def finalize(self):
        """ Train the LightGBM model with the data and value stored."""
        target = np.array(self.targets).flatten()
        # TODO: Set hp for parameters below.
        self.lgbm.set_params(**self.get_params())
        self.lgbm.fit(X=np.asarray(self.data), y=target)
        self.data = []
        self.targets = []

    @property
    def output_shape(self):
        return self.y_shape

    def set_weights(self, weights):
        self.lgbm = weights['lgbm']
        self.y_shape = weights['y_shape']


class LightGBMClassifier(LightGBMModel):
    """Collect data, train and test the LightGBM for classification task.

    Input data are np.array etc. np.random.rand(example_number, feature_number).
    Input labels are encoded labels in np.array form
    etc. np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).
    Outputs are predicted encoded labels in np.array form.

    The instance of this LightGBMClassifier class must be followed by
    an IdentityBlock and an EmptyHead as shown in LightGBMClassifierBlock class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lgbm = lgb.LGBMClassifier()
        self._one_hot_encoder = utils.OneHotEncoder()

    def update(self, x, y=None):
        """ Store the train data and decode.

        # Arguments
            x: Eager Tensor. The data to be stored.
            y: Eager Tensor. The label to be stored.
        """
        y = nest.flatten(y)[0].numpy()
        self.data.append(nest.flatten(x)[0].numpy())
        self._one_hot_encoder.fit_with_one_hot_encoded(np.array(y))
        self.y_shape = np.shape(y)
        y = y.reshape(1, -1)
        self.targets.append(nest.flatten(self._one_hot_encoder.decode(y))[0])

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
        ypred = super().transform(x, fit)
        y = self._one_hot_encoder.encode(ypred)
        y = y.reshape((-1))
        return y

    def output_types(self):
        return tf.int32,

    def set_weights(self, weights):
        super().set_weights(weights)
        self._one_hot_encoder = weights['_one_hot_encoder']

    def get_weights(self):
        return {'lgbm': self.lgbm,
                '_one_hot_encoder': self._one_hot_encoder,
                'y_shape': self.y_shape}

    def clear_weights(self):
        self.lgbm = lgb.LGBMClassifier()
        self._one_hot_encoder = utils.OneHotEncoder()
        self.y_shape = None


class LightGBMRegressor(LightGBMModel):
    """Collect data, train and test the LightGBM for regression task.

    Input data are np.array etc. np.random.rand(example_number, feature_number).
    Input value are real number in np.array form
    etc. np.array([1.1, 2.1, 4.2, 0.3, 2.4, 8.5, 7.3, 8.4, 9.4, 4.3]).
    Outputs are predicted value in np.array form.

    The instance of this LightGBMRegressor class must be followed by
    an IdentityBlock and an EmptyHead as shown in LightGBMRegressorBlock class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lgbm = lgb.LGBMRegressor()

    def update(self, x, y=None):
        """ Store the train data.

        # Arguments
            x: Eager Tensor. The data to be stored.
            y: Eager Tensor. The value to be stored.
        """
        self.data.append(nest.flatten(x)[0].numpy())
        self.y_shape = np.shape(y)
        self.targets.append(nest.flatten(y))

    def output_types(self):
        return tf.float64,

    def get_weights(self):
        return {'lgbm': self.lgbm,
                'y_shape': self.y_shape}

    def clear_weights(self):
        self.lgbm = lgb.LGBMRegressor()
        self.y_shape = None


class ImageAugmentation(Preprocessor):
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
        else:
            min_value = 1. - value
            max_value = 1. + value
        return min_value, max_value

    def transform(self, x, fit=False):
        if not fit:
            return x
        np.random.seed(self.seed)
        self.shape = x.shape
        target_height, target_width, channels = self.shape
        rotation_range = self.rotation_range
        k_choices = {}
        if rotation_range == 0:
            k_choices = [0]
        elif rotation_range == 90:
            k_choices = [0, 1, 3]
        elif rotation_range == 180:
            k_choices = [0, 1, 2, 3]
        x = tf.image.rot90(x, k=random.choice(k_choices))

        random_crop = self.random_crop
        if random_crop:
            crop_height = np.random.randint(low=1, high=target_height)
            crop_width = np.random.randint(low=1, high=target_width)
            crop_size = [crop_height,
                         crop_width,
                         channels]
            target_shape = (target_height, target_width)
            x = tf.image.resize(
                tf.image.random_crop(x, size=crop_size, seed=self.seed),
                size=target_shape)

        brightness_range = self.brightness_range
        if brightness_range != 0:
            x = tf.image.random_brightness(x, self.brightness_range, self.seed)

        saturation_range = self.saturation_range
        if saturation_range != 0:
            min_value, max_value = self.saturation_range
            x = tf.image.random_saturation(x, min_value, max_value, self.seed)

        contrast_range = self.contrast_range
        if contrast_range != 0:
            min_value, max_value = self.contrast_range
            x = tf.image.random_contrast(x, min_value, max_value, self.seed)

        translation = self.translation
        if translation:
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

        horizontal_flip = self.horizontal_flip
        if horizontal_flip:
            x = tf.image.flip_left_right(x)

        vertical_flip = self.horizontal_flip
        if vertical_flip:
            x = tf.image.flip_up_down(x)

        gaussian_noise = self.gaussian_noise
        if gaussian_noise:
            noise = tf.random.normal(shape=tf.shape(x),
                                     mean=0.0,
                                     stddev=1.0,
                                     seed=self.seed,
                                     dtype=tf.float32)
            x = tf.add(x, noise)
        return x

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self.inputs[0].shape

    def update(self, x, y=None):
        pass

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


def return_zero():
    return 0


class FeatureEngineering(Preprocessor):

    def __init__(self, column_types, max_columns=1000, **kwargs):
        super().__init__(**kwargs)
        self.column_types = column_types
        self.max_columns = max_columns
        self.num_columns = len(column_types)
        self.num_rows = 0
        self._shape = None
        # A list of categorical column indices.
        self.categorical_col = []
        # A list of numerical column indices.
        self.numerical_col = []
        self.label_encoders = {}
        self.value_counters = {}
        self.categorical_categorical = {}
        self.numerical_categorical = {}
        self.count_frequency = {}
        self.high_level1_col = []  # more than 32, less than 100
        self.high_level2_col = []  # more than 100
        self.high_level_cat_cat = {}
        self.high_level_num_cat = {}
        # debug
        self.transform_count = 0

        for index, column_type in enumerate(self.column_types):
            if column_type == 'categorical':
                self.categorical_col.append(index)
            elif column_type == 'numerical':
                self.numerical_col.append(index)
            else:
                raise ValueError('Unsupported column type: '
                                 '{type}'.format(type=column_type))

        for index, cat_col_index1 in enumerate(self.categorical_col):
            self.label_encoders[cat_col_index1] = utils.LabelEncoder()
            self.value_counters[cat_col_index1] = collections.defaultdict(
                return_zero)
            self.count_frequency[cat_col_index1] = {}
            for cat_col_index2 in self.categorical_col[index + 1:]:
                self.categorical_categorical[(
                    cat_col_index1,
                    cat_col_index2)] = collections.defaultdict(return_zero)
            for num_col_index in self.numerical_col:
                self.numerical_categorical[(
                    num_col_index,
                    cat_col_index1)] = collections.defaultdict(return_zero)

    def update(self, x, y=None):
        self.num_rows += 1
        x = nest.flatten(x)[0].numpy()
        for index in range(len(x)):
            x[index] = x[index].decode('utf-8')

        self._impute(x)

        for col_index in self.categorical_col:
            key = str(x[col_index])
            self.label_encoders[col_index].update(key)
            self.value_counters[col_index][key] += 1

        for col_index1, col_index2 in self.categorical_categorical.keys():
            key = (str(x[col_index1]), str(x[col_index2]))
            self.categorical_categorical[(col_index1, col_index2)][key] += 1

        for num_col_index, cat_col_index in self.numerical_categorical.keys():
            key = str(x[cat_col_index])
            v = x[num_col_index]
            self.numerical_categorical[(num_col_index, cat_col_index)][key] += v

    def transform(self, x, fit=False):
        x = nest.flatten(x)[0].numpy()
        # debug
        self.transform_count += 1
        # print('transforming # ' + repr(self.transform_count) + ' : ' + repr(x))
        for index in range(len(x)):
            x[index] = x[index].decode('utf-8')
        self._impute(x)

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
            pair = (str(x[col_index1]), str(x[col_index2]))
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
        self._shape = (self.num_rows, len(np.hstack((x, np.array(new_values)))))
        # debug
        # print('transform to ->->->')
        # print(np.hstack((x, np.array(new_values))))
        return np.hstack((x, np.array(new_values)))

    def _impute(self, x):
        for col_index in range(self.num_columns):
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
            # if num_cat <= 32:
            #     continue
            # calculate frequency
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

            # extract high level columns from num-cat dict and calculte mean
            for num_col_index in self.numerical_col:
                pair = (num_col_index, cat_col_index1)
                self.high_level_num_cat[pair] = self.numerical_categorical[pair]
                for key, value in self.high_level_num_cat[pair].items():
                    self.high_level_num_cat[pair][key] /= self.value_counters[
                        cat_col_index1][key]
        # debug
        print('column_num = ' + repr(self.num_columns))
        print('row_num = ' + repr(self.num_rows))
        print('categorical_col = ' + repr(self.categorical_col))
        print('numerical_col = ' + repr(self.numerical_col))
        print('label_encoders = ' + repr(self.label_encoders))
        print('value_counters = ' + repr(self.value_counters))
        print('categorical_categorical = ' + repr(self.categorical_categorical))
        print('numerical_categorical = ' + repr(self.numerical_categorical))
        print('count_frequency = ' + repr(self.count_frequency))
        print('high_level1_col = ' + repr(self.high_level1_col))
        print('high_level2_col = ' + repr(self.high_level2_col))
        print('high_level_cat_cat = ' + repr(self.high_level_cat_cat))
        print('high_level_num_cat = ' + repr(self.high_level_num_cat))

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self._shape

    def get_weights(self):
        return {'_shape': self._shape,
                'num_rows': self.num_rows,
                'categorical_col': self.categorical_col,
                'numerical_col': self.numerical_col,
                'label_encoders': self.label_encoders,
                'value_counters': self.value_counters,
                'categorical_categorical': self.categorical_categorical,
                'numerical_categorical': self.numerical_categorical,
                'count_frequency': self.count_frequency,
                'high_level1_col': self.high_level1_col,
                'high_level2_col': self.high_level2_col,
                'high_level_cat_cat': self.high_level_cat_cat,
                'high_level_num_cat': self.high_level_num_cat}

    def set_weights(self, weights):
        self._shape = weights['_shape']
        self.num_rows = weights['num_rows']
        self.categorical_col = weights['categorical_col']
        self.numerical_col = weights['numerical_col']
        self.label_encoders = weights['label_encoders']
        self.value_counters = weights['value_counters']
        self.categorical_categorical = weights['categorical_categorical']
        self.numerical_categorical = weights['numerical_categorical']
        self.count_frequency = weights['count_frequency']
        self.high_level1_col = weights['high_level1_col']
        self.high_level2_col = weights['high_level2_col']
        self.high_level_cat_cat = weights['high_level_cat_cat']
        self.high_level_num_cat = weights['high_level_num_cat']

    def clear_weights(self):
        self._shape = None
        self.num_rows = 0
        self.categorical_col = []
        self.numerical_col = []
        self.label_encoders = {}
        self.value_counters = {}
        self.categorical_categorical = {}
        self.numerical_categorical = {}
        self.count_frequency = {}
        self.high_level1_col = []  # less than 32
        self.high_level2_col = []  # more than 100
        self.high_level_cat_cat = {}
        self.high_level_num_cat = {}

    def get_config(self):
        return {'column_num': self.num_columns,
                'column_types': self.column_types,
                'max_columns': self.max_columns}

    def set_config(self, config):
        self.num_columns = config['column_num']
        self.column_types = config['column_types']
        self.max_columns = config['max_columns']
