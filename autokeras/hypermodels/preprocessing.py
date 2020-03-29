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
        translation: Boolean. Whether to translate the image. Defaults to True.
        horizontal_flip: Boolean. Whether to flip the image horizontally.
            Defaults to True.
        vertical_flip: Boolean. Whether to flip the image vertically.
            Defaults to True.
        rotation_range: A positive float represented as fraction of 2pi, or a tuple
            of size 2 representing lower and upper bound for rotating clockwise and
            counter-clockwise. When represented as a single float, lower = upper.
            Defaults to 0.5.
        random_crop: Boolean. Whether to crop the image randomly. Default to True.
        zoom_range: Boolean. A positive float represented as fraction value, or a
            tuple of 2 representing fraction for zooming horizontally and vertically.
            For instance, `zoom_range=0.2` result in a random zoom range from 80% to
            120%. Defaults to 0.5.
        contrast_range: A positive float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound. When represented as a
            single float, lower = upper. The contrast factor will be randomly picked
            between [1.0 - lower, 1.0 + upper]. Defaults to 0.5.
    """

    def __init__(self,
                 translation=True,
                 horizontal_flip=True,
                 vertical_flip=True,
                 rotation_range=0.5,
                 random_crop=True,
                 zoom_range=0.5,
                 contrast_range=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.translation = translation,
        self.horizontal_flip = horizontal_flip,
        self.vertical_flip = vertical_flip,
        self.rotation_range = rotation_range,
        self.random_crop = random_crop,
        self.zoom_range = zoom_range,
        self.contrast_range = contrast_range,
        self.shape = None

    def build(self, hp, inputs=None):
        list([
            preprocessing.RandomContrast,
            preprocessing.RandomTranslation,
            preprocessing.RandomFlip,
            preprocessing.RandomRotation,
            preprocessing.RandomZoom
        ])
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
