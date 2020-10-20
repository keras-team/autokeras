# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from typing import Tuple
from typing import Union

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import analysers
from autokeras import keras_layers
from autokeras.engine import block as block_module


class Normalization(block_module.Block):
    """Perform basic image transformation and augmentation.

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
        config.update({"axis": self.axis})
        return config


class TextToIntSequence(block_module.Block):
    """Convert raw texts to sequences of word indices.

    # Arguments
        output_sequence_length: Int. The maximum length of a sentence. If
            unspecified, it would be tuned automatically.
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
    """

    def __init__(
        self,
        output_sequence_length: Optional[int] = None,
        max_tokens: int = 20000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_sequence_length = output_sequence_length
        self.max_tokens = max_tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_sequence_length": self.output_sequence_length,
                "max_tokens": self.max_tokens,
            }
        )
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        if self.output_sequence_length is not None:
            output_sequence_length = self.output_sequence_length
        else:
            output_sequence_length = hp.Choice(
                "output_sequence_length", [64, 128, 256, 512], default=64
            )
        output_node = preprocessing.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=output_sequence_length,
        )(input_node)
        return output_node


class TextToNgramVector(block_module.Block):
    """Convert raw texts to n-gram vectors.

    # Arguments
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
        ngrams: Int or tuple of ints. Passing an integer will create ngrams up to
            that integer, and passing a tuple of integers will create ngrams for the
            specified values in the tuple. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        max_tokens: int = 20000,
        ngrams: Union[int, Tuple[int], None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.ngrams = ngrams

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        if self.ngrams is not None:
            ngrams = self.ngrams
        else:
            ngrams = hp.Int("ngrams", min_value=1, max_value=2, default=2)
        return preprocessing.TextVectorization(
            max_tokens=self.max_tokens, ngrams=ngrams, output_mode="tf-idf"
        )(input_node)

    def get_config(self):
        config = super().get_config()
        config.update({"max_tokens": self.max_tokens, "ngrams": self.ngrams})
        return config


class ImageAugmentation(block_module.Block):
    """Collection of various image augmentation methods.

    # Arguments
        translation_factor: A positive float represented as fraction value, or a
            tuple of 2 representing fraction for translation vertically and
            horizontally.  For instance, `translation_factor=0.2` result in a random
            translation factor within 20% of the width and height.
            If left unspecified, it will be tuned automatically.
        vertical_flip: Boolean. Whether to flip the image vertically.
            If left unspecified, it will be tuned automatically.
        horizontal_flip: Boolean. Whether to flip the image horizontally.
            If left unspecified, it will be tuned automatically.
        rotation_factor: Float. A positive float represented as fraction of 2pi
            upper bound for rotating clockwise and counter-clockwise. When
            represented as a single float, lower = upper.
            If left unspecified, it will be tuned automatically.
        zoom_factor: A positive float represented as fraction value, or a tuple of 2
            representing fraction for zooming vertically and horizontally. For
            instance, `zoom_factor=0.2` result in a random zoom factor from 80% to
            120%. If left unspecified, it will be tuned automatically.
        contrast_factor: A positive float represented as fraction of value, or a
            tuple of size 2 representing lower and upper bound. When represented as a
            single float, lower = upper. The contrast factor will be randomly picked
            between [1.0 - lower, 1.0 + upper]. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        translation_factor: Optional[Union[float, Tuple[float, float]]] = None,
        vertical_flip: Optional[bool] = None,
        horizontal_flip: Optional[bool] = None,
        rotation_factor: Optional[float] = None,
        zoom_factor: Optional[Union[float, Tuple[float, float]]] = None,
        contrast_factor: Optional[Union[float, Tuple[float, float]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.translation_factor = translation_factor
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.contrast_factor = contrast_factor

    @staticmethod
    def _get_fraction_value(value):
        if isinstance(value, tuple):
            return value
        return value, value

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node

        # Translate
        translation_factor = self.translation_factor
        if translation_factor is None:
            translation_factor = hp.Choice("translation_factor", [0.0, 0.1])
        if translation_factor != 0 and translation_factor != (0, 0):
            height_factor, width_factor = self._get_fraction_value(
                translation_factor
            )
            output_node = preprocessing.RandomTranslation(
                height_factor, width_factor
            )(output_node)

        # Flip
        horizontal_flip = self.horizontal_flip
        if horizontal_flip is None:
            horizontal_flip = hp.Boolean("horizontal_flip", default=True)
        vertical_flip = self.vertical_flip
        if self.vertical_flip is None:
            vertical_flip = hp.Boolean("vertical_flip", default=True)
        if not horizontal_flip and not vertical_flip:
            flip_mode = ""
        elif horizontal_flip and vertical_flip:
            flip_mode = "horizontal_and_vertical"
        elif horizontal_flip and not vertical_flip:
            flip_mode = "horizontal"
        elif not horizontal_flip and vertical_flip:
            flip_mode = "vertical"
        if flip_mode != "":
            output_node = preprocessing.RandomFlip(mode=flip_mode)(output_node)

        # Rotate
        rotation_factor = self.rotation_factor
        if rotation_factor is None:
            rotation_factor = hp.Choice("rotation_factor", [0.0, 0.1])
        if rotation_factor != 0:
            output_node = preprocessing.RandomRotation(rotation_factor)(output_node)

        # Zoom
        zoom_factor = self.zoom_factor
        if zoom_factor is None:
            zoom_factor = hp.Choice("zoom_factor", [0.0, 0.1])
        if zoom_factor != 0 and zoom_factor != (0, 0):
            height_factor, width_factor = self._get_fraction_value(zoom_factor)
            # TODO: Add back RandomZoom when it is ready.
            # output_node = preprocessing.RandomZoom(
            # height_factor, width_factor)(output_node)

        # Contrast
        contrast_factor = self.contrast_factor
        if contrast_factor is None:
            contrast_factor = hp.Choice("contrast_factor", [0.0, 0.1])
        if contrast_factor != 0 and contrast_factor != (0, 0):
            output_node = preprocessing.RandomContrast(contrast_factor)(output_node)

        return output_node

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "translation_factor": self.translation_factor,
                "horizontal_flip": self.horizontal_flip,
                "vertical_flip": self.vertical_flip,
                "rotation_factor": self.rotation_factor,
                "zoom_factor": self.zoom_factor,
                "contrast_factor": self.contrast_factor,
            }
        )
        return config


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
            if column_type == analysers.CATEGORICAL:
                # TODO: Search to use one-hot or int.
                encoding.append(keras_layers.INT)
            else:
                encoding.append(keras_layers.NONE)
        return keras_layers.MultiCategoryEncoding(encoding)(input_node)

    @classmethod
    def from_config(cls, config):
        column_types = config.pop("column_types")
        column_names = config.pop("column_names")
        instance = cls(**config)
        instance.column_types = column_types
        instance.column_names = column_names
        return instance

    def get_config(self):
        config = super().get_config()
        config.update(
            {"column_types": self.column_types, "column_names": self.column_names}
        )
        return config
