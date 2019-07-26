import tensorflow as tf
import numpy as np
from sklearn.feature_extraction import text
from sklearn import feature_selection
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


class ImageAugment(HyperPreprocessor):

    def __init__(self,
                 rotation_range=None,
                 whether_random_crop=None,
                 brightness_range=None,  # fraction 0-1  [X]
                 saturation_range=None,  # fraction 0-1  [X]
                 contrast_range=None,  # fraction 0-1  [X]
                 horizontal_flip=None,  # boolean  [X]
                 vertical_flip=None,
                 whether_translation=None,
                 gaussian_noise=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rotation_range = rotation_range
        self.whether_random_crop = whether_random_crop
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.whether_translation = whether_translation
        self.gaussian_noise = gaussian_noise
        
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
        return min_value, max_value

    def transform(self, x):
        self._shape = x.shape
        target_height, target_width, channels = self._shape
        rotation_range = self.rotation_range
        if rotation_range is None:
            rotation_range = self._hp.Choice('rotation_range',
                                             [90, 180, 270, 0],
                                             default=90)
        whether_random_crop = self.whether_random_crop
        if whether_random_crop is None:
            whether_random_crop = self._hp.Choice('whether_random_crop',
                                                  [True, False],
                                                  default=True)
        brightness_range = self.brightness_range
        if brightness_range is None:
            brightness_range = self._hp.Range('brightness_range', 0, 1)
        saturation_range = self.saturation_range
        if saturation_range is None:
            saturation_range = self._hp.Range('saturation_range', 0, 1)
        contrast_range = self.contrast_range
        if contrast_range is None:
            contrast_range = self._hp.Range('contrast_range', 0, 1)
        horizontal_flip = self.horizontal_flip
        if horizontal_flip is None:
            horizontal_flip = self._hp.Choice('horizontal_flip',
                                              [True, False],
                                              default=True)
        vertical_flip = self.vertical_flip
        if vertical_flip is None:
            vertical_flip = self._hp.Choice('vertical_flip',
                                            [True, False],
                                            default=True)
        whether_translation = self.whether_translation
        if whether_translation is None:
            whether_translation = self._hp.Choice('whether_translation',
                                                  [True, False],
                                                  default=True)
        gaussian_noise = self.gaussian_noise
        if gaussian_noise is None:
            gaussian_noise = self._hp.Choice('gaussian_noise',
                                             [True, False],
                                             default=True)
        x = tf.cast(x, dtype=tf.float32)
        if gaussian_noise:
            noise = tf.random_normal(shape=tf.shape(x),
                                     mean=0.0, stddev=1.0, dtype=tf.float32)
            x = tf.add(x, noise)
        '''if whether_translation:
            x = tf.image.pad_to_bounding_box(x, self._hp.Choice('translation_top'),
                                             self._hp.Choice('translation_left'),
                                             self._hp.Choice('target_height') +
                                             self._hp.Choice('translation_bottom') +
                                             self._hp.Choice('translation_top'),
                                             self._hp.Choice('target_width') +
                                             self._hp.Choice('translation_right') +
                                             self._hp.Choice('translation_left'))
            x = tf.image.crop_to_bounding_box(x,
                                              self._hp.Choice('translation_bottom'),
                                              self._hp.Choice('translation_right'),
                                              self._hp.Choice('target_height'),
                                              self._hp.Choice('target_width'))'''
        if rotation_range:
            if rotation_range == 90:
                x = tf.image.rot90(x, k=1)
            elif rotation_range == 180:
                x = tf.image.rot90(x, k=2)
            elif rotation_range == 270:
                x = tf.image.rot90(x, k=3)
            else:
                x = tf.image.rot90(x, k=4)
        if brightness_range:
            min_value, max_value = self.__get_min_and_max(
                brightness_range,
                'brightness_range')
            x = tf.image.random_brightness(x, min_value, max_value)
        if saturation_range:
            min_value, max_value = self.__get_min_and_max(
                saturation_range,
                'saturation_range')
            x = tf.image.random_saturation(x, min_value, max_value)
        if contrast_range:
            min_value, max_value = self.__get_min_and_max(
                contrast_range,
                'contrast_range')
            x = tf.image.random_contrast(x, min_value, max_value)
        if whether_random_crop:
            crop_size = [self._hp.Choice('random_crop_height'),
                         self._hp.Choice('random_crop_width'),
                         channels]
            seed = np.random.randint(self._hp.Choice('random_crop_seed'))
            target_shape = (target_height, target_width)
            x = tf.image.resize(
                tf.image.random_crop(x, size=crop_size, seed=seed),
                size=target_shape)
        if horizontal_flip:
            x = tf.image.flip_left_right(x)
        if vertical_flip:
            x = tf.image.flip_up_down(x)
        return x

    def output_types(self):
        return tf.float32

    def output_shape(self):
        return self._shape

    def update(self, x):
        pass
