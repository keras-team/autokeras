from typing import Optional

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
        translation_factor: A positive float represented as fraction value, or a
            tuple of 2 representing fraction for translation vertically and
            horizontally.  For instance, `translation_factor=0.2` result in a random
            translation factor within 20% of the width and height. Defaults to 0.5.
        vertical_flip: Boolean. Whether to flip the image vertically.
            If left unspecified, it will be tuned automatically.
        horizontal_flip: Boolean. Whether to flip the image horizontally.
            If left unspecified, it will be tuned automatically.
        rotation_factor: Float. A positive float represented as fraction of 2pi
            upper bound for rotating clockwise and counter-clockwise. When
            represented as a single float, lower = upper.  Defaults to 0.5.
        zoom_factor: A positive float represented as fraction value, or a tuple of 2
            representing fraction for zooming vertically and horizontally. For
            instance, `zoom_factor=0.2` result in a random zoom factor from 80% to
            120%. Defaults to 0.5.
        contrast_factor: A positive float represented as fraction of value, or a
            tuple of size 2 representing lower and upper bound. When represented as a
            single float, lower = upper. The contrast factor will be randomly picked
            between [1.0 - lower, 1.0 + upper]. Defaults to 0.5.
    """

    def __init__(self,
                 translation_factor=0.5,
                 vertical_flip=None,
                 horizontal_flip=None,
                 rotation_factor=0.5,
                 zoom_factor=0.5,
                 contrast_factor=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.translation_factor = translation_factor
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.contrast_factor = contrast_factor
        self.shape = None

    @staticmethod
    def _get_fraction_value(value):
        if isinstance(value, tuple):
            return value
        return value, value

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node

        if self.translation_factor != 0 and self.translation_factor != (0, 0):
            height_factor, width_factor = self._get_fraction_value(
                self.translation_factor)
            output_node = preprocessing.RandomTranslation(
                height_factor, width_factor)(output_node)

        horizontal_flip = self.horizontal_flip
        if horizontal_flip is None:
            horizontal_flip = hp.Boolean('horizontal_flip', default=True)
        vertical_flip = self.vertical_flip
        if self.vertical_flip is None:
            vertical_flip = hp.Boolean('vertical_flip', default=True)
        if not horizontal_flip and not vertical_flip:
            flip_mode = ''
        elif horizontal_flip and vertical_flip:
            flip_mode = 'horizontal_and_vertical'
        elif horizontal_flip and not vertical_flip:
            flip_mode = 'horizontal'
        elif not horizontal_flip and vertical_flip:
            flip_mode = 'vertical'
        if flip_mode != '':
            output_node = preprocessing.RandomFlip(
                mode=flip_mode)(output_node)

        if self.rotation_factor != 0:
            output_node = preprocessing.RandomRotation(
                self.rotation_factor)(output_node)

        if self.zoom_factor != 0 and self.zoom_factor != (0, 0):
            height_factor, width_factor = self._get_fraction_value(
                self.zoom_factor)
            output_node = preprocessing.RandomZoom(
                height_factor, width_factor)(output_node)

        if self.contrast_factor != 0 and self.contrast_factor != (0, 0):
            output_node = preprocessing.RandomContrast(
                self.contrast_factor)(output_node)

        return output_node


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
