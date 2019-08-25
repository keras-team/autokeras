import numpy as np
import tensorflow as tf
import random
from sklearn import feature_selection
from sklearn.feature_extraction import text
from tensorflow.python.util import nest

from autokeras import const
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
        self._shape = None

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
        return (min_value, max_value)

    def transform(self, x, fit=False):
        if not fit:
            return x
        np.random.seed(self.seed)
        self._shape = x.shape
        target_height, target_width, channels = self._shape
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

    def get_state(self):
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
                '_shape': self._shape}

    def set_state(self, state):
        self.rotation_range = state['rotation_range']
        self.random_crop = state['random_crop']
        self.brightness_range = state['brightness_range']
        self.saturation_range = state['saturation_range']
        self.contrast_range = state['contrast_range']
        self.translation = state['translation']
        self.horizontal_flip = state['horizontal_flip']
        self.vertical_flip = state['vertical_flip']
        self.gaussian_noise = state['gaussian_noise']
        self.seed = state['seed']
        self._shape = state['_shape']


class FeatureEngineering(Preprocessor):
    # TODO:
    def __init__(self, column_types, max_columns=1000, **kwargs):
        super().__init__(**kwargs)
        self.column_num = len(column_types)
        self.row_num = 0
        self._shape = None
        self.categorical_col = []
        self.numerical_col = []
        self.categorical_to_int_label = {}
        self.each_cat_num = {}
        self.total_cat_num = {}
        self.categorical_categorical = {}
        self.numerical_categorical = {}
        self.count_frequency = {}
        self.count_num_cat_mean = {}
        self.high_level1 = 32
        self.high_level2 = 100
        self.high_level1_col = []  # less than 32
        self.high_level2_col = []  # more than 100
        self.high_level_cat_cat = {}
        self.high_level_num_cat = {}

        for col in range(self.column_num):
            if column_types[col] == 'categorical':
                self.categorical_col.append(col)
            else:
                self.numerical_col.append(col)

        for i, cat_col_index1 in enumerate(self.categorical_col):
            self.categorical_to_int_label[cat_col_index1] = {}
            self.each_cat_num[cat_col_index1] = {}
            self.count_frequency[cat_col_index1] = {}
            for cat_col_index2 in self.categorical_col[i+1:]:
                self.categorical_categorical[(cat_col_index1, cat_col_index2)] = {}
            for num_col_index in self.numerical_col:
                self.numerical_categorical[(num_col_index, cat_col_index1)] = {}
                self.count_num_cat_mean[(num_col_index, cat_col_index1)] = {}

    def update(self, x, y=None):
        # debug
        print('updating row# : ' + repr(self.row_num))
        self.row_num += 1
        x = nest.flatten(x)[0].numpy()

        for col_index in range(self.column_num):
            x[col_index] = x[col_index].decode('utf-8')
            if col_index in self.numerical_col:
                if x[col_index] == 'nan':
                    x[col_index] = 0.0
                else:
                    x[col_index] = float(x[col_index])
            else:
                if x[col_index] == 'nan':
                    x[col_index] = 0

        for col_index in self.categorical_col:
            key = str(x[col_index])
            if key in self.categorical_to_int_label[col_index]:
                self.each_cat_num[col_index][key] += 1
            else:
                new_value = len(self.categorical_to_int_label[col_index])
                self.categorical_to_int_label[col_index][key] = new_value
                self.each_cat_num[col_index][key] = 1

        for col_index1, col_index2 in self.categorical_categorical.keys():
            key = (str(x[col_index1]), str(x[col_index2]))
            if key in self.categorical_categorical[(col_index1, col_index2)]:
                self.categorical_categorical[(col_index1, col_index2)][key] += 1
            else:
                self.categorical_categorical[(col_index1, col_index2)][key] = 1

        for num_col_index, cat_col_index in self.numerical_categorical.keys():
            key = str(x[cat_col_index])
            v = x[num_col_index]
            if key in self.numerical_categorical[(num_col_index, cat_col_index)]:
                self.numerical_categorical[(num_col_index, cat_col_index)][key] += v
            else:
                self.numerical_categorical[(num_col_index, cat_col_index)][key] = v
        # # debug
        # print('column_num = ' + repr(self.column_num))
        # print('row_num = ' + repr(self.row_num))
        # print('categorical_col = ' + repr(self.categorical_col))
        # print('numerical_col = ' + repr(self.numerical_col))
        # print('categorical_to_int_label = ' + repr(self.categorical_to_int_label))
        # print('each_cat_num = ' + repr(self.each_cat_num))
        # print('total_cat_num = ' + repr(self.total_cat_num))
        # print('categorical_categorical = ' + repr(self.categorical_categorical))
        # print('numerical_categorical = ' + repr(self.numerical_categorical))
        # print('count_frequency = ' + repr(self.count_frequency))
        # print('count_num_cat_mean = ' + repr(self.count_num_cat_mean))
        # print('high_level1_col = ' + repr(self.high_level1_col))
        # print('high_level2_col = ' + repr(self.high_level2_col))
        # print('high_level_cat_cat = ' + repr(self.high_level_cat_cat))
        # print('high_level_num_cat = ' + repr(self.high_level_num_cat))

    def transform(self, x, fit=False):

        x = nest.flatten(x)[0].numpy()
        # debug
        print('transforming data : ' + repr(x))
        for col_index in range(self.column_num):
            x[col_index] = x[col_index].decode('utf-8')
            if col_index in self.numerical_col:
                if x[col_index] == 'nan':
                    x[col_index] = 0.0
                else:
                    x[col_index] = float(x[col_index])
            else:
                if x[col_index] == 'nan':
                    x[col_index] = 0

        add_column = []
        # append frequency
        for col_index in self.high_level1_col:
            cat_name = str(x[col_index])
            add_column.append(self.count_frequency[col_index][cat_name])

        # append cat-cat value
        for key, value in self.high_level_cat_cat.items():
            col_index1, col_index2 = key
            pair = (str(x[col_index1]), str(x[col_index2]))
            add_column.append(value[pair])

        # append num-cat value
        for key, value in self.high_level_num_cat.items():
            num_col_index, cat_col_index = key
            cat_name = str(x[cat_col_index])
            add_column.append(value[cat_name])
        add_column_array = np.array(add_column)

        # LabelEncoding
        for col_index in self.categorical_col:
            key = str(x[col_index])
            x[col_index] = self.categorical_to_int_label[col_index][key]

        self._shape = (self.row_num, len(np.hstack((x, add_column_array))))
        print('transform to ->->-> ')
        print(np.hstack((x, add_column_array)))
        return np.hstack((x, add_column_array))

    def finalize(self):
        # debug
        print('finalizing...')
        # divide column according to category number of the column
        for col_index in self.each_cat_num.keys():
            cat_num = len(self.each_cat_num[col_index])
            self.total_cat_num[col_index] = cat_num
            # debug
            print('col_index is '+repr(col_index)+' and cat_num is ' + repr(cat_num))
            if cat_num <= self.high_level1:
                continue
            # calculate frequency 
            elif cat_num <= self.high_level2:
                self.high_level1_col.append(col_index)
                for key, value in self.each_cat_num[col_index].items():
                    self.count_frequency[col_index][key] = \
                                                        value/(self.row_num*cat_num)
            else:
                self.high_level2_col.append(col_index)

        self.high_level2_col.sort()
        # debug
        print('high_level1_col = ' + repr(self.high_level1_col))
        print('high_level2_col = ' + repr(self.high_level2_col))

        for i, cat_col_index1 in enumerate(self.high_level2_col):
            # extract high level columns from cat-cat dict
            for cat_col_index2 in self.high_level2_col[i+1:]:
                pair = (cat_col_index1, cat_col_index2)
                self.high_level_cat_cat[pair] = self.categorical_categorical[pair]

            # extract high level columns from num-cat dict and calculte mean
            for num_col_index in self.numerical_col:
                pair = (num_col_index, cat_col_index1)
                self.high_level_num_cat[pair] = self.numerical_categorical[pair]
                for key, value in self.high_level_num_cat[pair].items():
                    new_value = value/self.each_cat_num[cat_col_index1][key]
                    self.high_level_num_cat[pair][key] = new_value
        # debug
        print('column_num = ' + repr(self.column_num))
        print('row_num = ' + repr(self.row_num))
        print('categorical_col = ' + repr(self.categorical_col))
        print('numerical_col = ' + repr(self.numerical_col))
        print('categorical_to_int_label = ' + repr(self.categorical_to_int_label))
        print('each_cat_num = ' + repr(self.each_cat_num))
        print('total_cat_num = ' + repr(self.total_cat_num))
        print('categorical_categorical = ' + repr(self.categorical_categorical))
        print('numerical_categorical = ' + repr(self.numerical_categorical))
        print('count_frequency = ' + repr(self.count_frequency))
        print('count_num_cat_mean = ' + repr(self.count_num_cat_mean))
        print('high_level1_col = ' + repr(self.high_level1_col))
        print('high_level2_col = ' + repr(self.high_level2_col))
        print('high_level_cat_cat = ' + repr(self.high_level_cat_cat))
        print('high_level_num_cat = ' + repr(self.high_level_num_cat))

    def output_types(self):
        return tf.float64,

    @property
    def output_shape(self):
        return self._shape

    def get_state(self):
        return {'column_num': self.column_num,
                'row_num': self.row_num,
                '_shape': self._shape,
                'categorical_col': self.categorical_col,
                'numerical_col': self.numerical_col,
                'categorical_to_int_label': self.categorical_to_int_label,
                'each_cat_num': self.each_cat_num,
                'total_cat_num': self.total_cat_num,
                'categorical_categorical': self.categorical_categorical,
                'numerical_categorical': self.numerical_categorical,
                'count_frequency': self.count_frequency,
                'count_num_cat_mean': self.count_num_cat_mean,
                'high_level1': self.high_level1,
                'high_level2': self.high_level2,
                'high_level1_col': self.high_level1_col,
                'high_level2_col': self.high_level2_col,
                'high_level_cat_cat': self.high_level_cat_cat,
                'high_level_num_cat': self.high_level_num_cat}

    def set_state(self, state):
        self.column_num = state['column_num']
        self.row_num = state['row_num']
        self._shape = state['_shape']
        self.categorical_col = state['categorical_col']
        self.numerical_col = state['numerical_col']
        self.categorical_to_int_label = state['categorical_to_int_label']
        self.each_cat_num = state['each_cat_num']
        self.total_cat_num = state['total_cat_num']
        self.categorical_categorical = state['categorical_categorical']
        self.numerical_categorical = state['numerical_categorical']
        self.count_frequency = state['count_frequency']
        self.count_num_cat_mean = state['count_num_cat_mean']
        self.high_level1 = 32
        self.high_level2 = 100
        self.high_level1_col = state['high_level1_col']
        self.high_level2_col = state['high_level2_col']
        self.high_level_cat_cat = state['high_level_cat_cat']
        self.high_level_num_cat = state['high_level_num_cat']
