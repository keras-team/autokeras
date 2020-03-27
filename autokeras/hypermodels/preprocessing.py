from typing import Optional

import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import keras_layers
from autokeras.engine import block as block_module


class Normalization(block_module.Block):
    """ Perform basic image transformation and augmentation.

    # Arguments
        axis: Integer or tuple of integers, the axis or axes that should be
            normalized (typically the features axis). We will normalize each element
            in the specified axis. The default is '-1' (the innermost axis); 0 (the
            batch axis) is not allowed.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        return preprocessing.Normalization(axis=self.axis)(input_node)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


class TextToIntSequence(block_module.Block):
    """Convert raw texts to sequences of word indices.

    # Arguments
        output_sequence_length: Int. The maximum length of a sentence. If
            unspecified, it would be tuned automatically.
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
    """

    def __init__(self,
                 output_sequence_length: Optional[int] = None,
                 max_tokens: int = 20000,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_sequence_length = output_sequence_length
        self.max_tokens = max_tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_sequence_length': self.output_sequence_length,
            'max_tokens': self.max_tokens,
        })
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        if self.output_sequence_length is not None:
            output_sequence_length = self.output_sequence_length
        else:
            output_sequence_length = hp.Choice('output_sequence_length',
                                               [64, 128, 256, 512], default=64)
        output_node = preprocessing.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode='int',
            output_sequence_length=output_sequence_length)(input_node)
        return output_node


class TextToNgramVector(block_module.Block):
    """Convert raw texts to n-gram vectors.

    # Arguments
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
    """

    def __init__(self,
                 max_tokens=20000,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        return preprocessing.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode='tf-idf')(input_node)

    def get_config(self):
        config = super().get_config()
        config.update({'max_tokens': self.max_tokens})
        return config


class ImageAugmentation(block_module.Block):
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

    def build(self, hp, inputs=None):
        return inputs


class CategoricalToNumerical(block_module.Block):
    """Encode the categorical features to numerical features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.column_types = None
        self.column_names = None

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        encoding = []
        for column_name in self.column_names:
            column_type = self.column_types[column_name]
            if column_type == adapters.CATEGORICAL:
                # TODO: Search to use one-hot or int.
                encoding.append(keras_layers.INT)
            else:
                encoding.append(keras_layers.NONE)
        return keras_layers.CategoricalEncoding(encoding)(input_node)
