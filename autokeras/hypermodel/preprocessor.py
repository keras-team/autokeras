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

    def update(self, x):
        """Incrementally fit the preprocessor with a single training instance.

        # Arguments
            x: EagerTensor. A single instance in the training dataset.
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

    def update(self, x):
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
        random_crop: Boolean. Whether crop the image randomly. Default to True.
        brightness_range: Positive int.
            Serve as 'max_delta' in tf.image.random_brightness. Default to 0.5.
        saturation_range: Positive int or Tuple.
            Serve as the bound of the tf.image.random_saturation. Default to 0.5.
        contrast_range: Positive int or Tuple.
            Serve as the bound of the tf.image.random_contrast. Default to 0.5.
        translation: Boolean. Whether translate the image.
        horizontal_flip: Boolean. Whether flip the image horizontally.
        vertical_flip: Boolean. Whether flip the image vertically.
        gaussian_noise: Boolean. Whether add some gaussian noise to the image.
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
            pad_top = np.random.randint(low=0, high=target_height)
            pad_left = np.random.randint(low=0, high=target_width)
            pad_bottom = np.random.randint(low=0, high=target_height)
            pad_right = np.random.randint(low=0, high=target_width)
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

    def update(self, x):
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
