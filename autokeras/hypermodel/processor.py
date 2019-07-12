import tensorflow as tf
import numpy as np

from autokeras import utils
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

    def fit(self, hp, data):
        shape = utils.dataset_shape(data)
        axis = tuple(range(len(shape) - 1))

        def sum_up(old_state, new_elem):
            return old_state + new_elem

        def sum_up_square(old_state, new_elem):
            return old_state + tf.square(new_elem)

        num_instance = data.reduce(np.float64(0), lambda x, _: x + 1)
        total_sum = data.reduce(np.float64(0), sum_up) / num_instance
        self.mean = tf.reduce_mean(total_sum, axis=axis)

        total_sum_square = data.reduce(np.float64(0), sum_up_square) / num_instance
        square_mean = tf.reduce_mean(total_sum_square, axis=axis)
        self.std = tf.sqrt(square_mean - tf.square(self.mean))

    def transform(self, hp, data):
        """ Transform the test data, perform normalization.

        # Arguments
            data: Tensorflow Dataset. The data to be transformed.

        # Returns
            A DataLoader instance.
        """
        # channel-wise normalize the image
        def normalize(x):
            return (x - self.mean) / self.std
        return data.map(normalize)

    
class ImageAugment(HyperPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = True
        self.length_dim = 0
        self.batch_num = 0
        self.target_height = 0
        self.target_width = 0
        self.channels = 0

    def fit(self, hp, inputs):
        if self.first:
            self.first = False
            inputs = tf.convert_to_tensor(inputs)
            self.length_dim = len(inputs.shape)
            if self.length_dim != 4:
                raise ValueError(
                    'The input of x_train should be a [batch_size, height, '
                    'width, channel] '
                    'shape tensor or list, but we get %s' % inputs.shape)
            self.batch_num, self.target_height, \
            self.target_width, self.channels = inputs.shape
            # TODO: Set the arguments if user didn't set
            return inputs
        else:
            pass

    @staticmethod
    def __get_min_and_max(value, name):
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

    @staticmethod
    def guassian_noise(image):
        noise = tf.random.normal(shape=tf.shape(image),
                                 mean=0.0, stddev=1.0, dtype=tf.float32)
        image = tf.add(image, noise)
        return image

    @staticmethod
    def translation(image, hp):
        x = tf.image.pad_to_bounding_box(image, hp.Choice('translation_top'),
                                         hp.Choice('translation_left'),
                                         hp.Choice('target_height') +
                                         hp.Choice('translation_bottom') +
                                         hp.Choice('translation_top'),
                                         hp.Choice('target_width') +
                                         hp.Choice('translation_right') +
                                         hp.Choice('translation_left'))
        image = tf.image.crop_to_bounding_box(x, hp.Choice('translation_bottom'),
                                              hp.Choice('translation_right'),
                                              hp.Choice('target_height'),
                                              hp.Choice('target_width'))
        return image

    @staticmethod
    def rotation(image, hp):
        if hp.Choice('rotation_range') == 90:
            image = tf.image.rot90(image, k=1)
        elif hp.Choice('rotation_range') == 180:
            image = tf.image.rot90(image, k=2)
        elif hp.Choice('rotation_range') == 270:
            image = tf.image.rot90(image, k=3)
        else:
            image = tf.image.rot90(image, k=4)
        return image

    def brightness(self, image, hp):
        min_value, max_value = self.__get_min_and_max(
            hp.Choice('brightness_range'),
            'brightness_range')
        image = tf.image.random_brightness(image, min_value, max_value)
        return image

    def saturation(self, image, hp):
        min_value, max_value = self.__get_min_and_max(
            hp.Choice('saturation_range'),
            'saturation_range')
        image = tf.image.random_saturation(image, min_value, max_value)
        return image

    def contrast(self, image, hp):
        min_value, max_value = self.__get_min_and_max(
            hp.Choice('contrast_range'),
            'contrast_range')
        image = tf.image.random_contrast(
            image, min_value, max_value)
        return image

    @staticmethod
    def random_crop(image, hp, batch_num, channels, 
                    target_height, target_width, random_crop_seed):
        crop_size = [batch_num, hp.Choice('random_crop_height'),
                     hp.Choice('random_crop_width'), channels]
        seed = np.random.randint(random_crop_seed)
        target_shape = (target_height, target_width)
        image = tf.image.resize(
            tf.image.random_crop(image, size=crop_size, seed=seed),
            size=target_shape)
        return image

    @staticmethod
    def horizontal_flip(image):
        image = tf.image.flip_left_right(image)
        return image

    @staticmethod
    def vertical_flip(image):
        image = tf.image.flip_up_down(image)
        return image

    def transform(self, hp, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.cast(inputs, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensors(inputs)
        if hp.Choice('gaussian_noise'):
            map_func = partial(self.guassian_noise)
            dataset.map(map_func=map_func)
        if hp.Choice('translation_bottom') or hp.Choice('translation_left') \
                or hp.Choice('translation_right') or hp.Choice('translation_top'):
            map_func = partial(self.translation, hp=hp)
            dataset.map(map_func=map_func)
        if hp.Choice('rotation_range'):
            map_func = partial(self.rotation, hp=hp)
            dataset.map(map_func=map_func)
        if hp.Choice('brightness_range'):
            map_func = partial(self.brightness, hp=hp)
            dataset.map(map_func=map_func)
        if hp.Choice('saturation'):
            map_func = partial(self.saturation, hp=hp)
            dataset.map(map_func=map_func)
        if hp.Choice('contrast_range'):
            map_func = partial(self.contrast, hp=hp)
            dataset.map(map_func=map_func)
        if hp.Choice('random_crop_height') and hp.Choice('random_crop_width'):
            map_func = partial(self.random_crop, hp=hp, batch_num=self.batch_num,
                               channels=self.channels, target_hight=self.target_height,
                               target_width=self.target_width, random_crop_seed=Constant.SEED)
            dataset.map(map_func=map_func)
        if hp.Choice('horizontal_crop'):
            map_func = self.horizontal_flip
            dataset.map(map_func=map_func)
        if hp.Choice('vertical_crop'):
            map_func = self.vertical_flip
            dataset.map(map_func=map_func)
        return dataset
