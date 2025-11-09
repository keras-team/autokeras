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

"""Blocks for data preprocessing.

They are built into keras preprocessing layers and will be part of the Keras
model.

"""
from typing import Optional
from typing import Tuple
from typing import Union

import keras
import tree
from keras import layers
from keras_tuner.engine import hyperparameters

from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import utils


class Normalization(block_module.Block):
    """Perform feature-wise normalization on data.

    Refer to Normalization layer in keras preprocessing layers for more
    information.

    # Arguments
        axis: Integer or tuple of integers, the axis or axes that should be
            normalized (typically the features axis). We will normalize each
            element in the specified axis. The default is '-1' (the innermost
            axis); 0 (the batch axis) is not allowed.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        return layers.Normalization(axis=self.axis)(input_node)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable(package="autokeras")
class ImageAugmentation(block_module.Block):
    """Collection of various image augmentation methods.

    # Arguments
        translation_factor: A positive float represented as fraction value, or a
            tuple of 2 representing fraction for translation vertically and
            horizontally, or a kerastuner.engine.hyperparameters.Choice range
            of positive floats. For instance, `translation_factor=0.2` result
            in a random translation factor within 20% of the width and height.
            If left unspecified, it will be tuned automatically.
        vertical_flip: Boolean. Whether to flip the image vertically.
            If left unspecified, it will be tuned automatically.
        horizontal_flip: Boolean. Whether to flip the image horizontally.
            If left unspecified, it will be tuned automatically.
        rotation_factor: Float or kerastuner.engine.hyperparameters.Choice range
            between [0, 1]. A positive float represented as fraction of 2pi
            upper bound for rotating clockwise and counter-clockwise. When
            represented as a single float, lower = upper.
            If left unspecified, it will be tuned automatically.
        zoom_factor: A positive float represented as fraction value, or a tuple
            of 2 representing fraction for zooming vertically and horizontally,
            or a kerastuner.engine.hyperparameters.Choice range of positive
            floats.  For instance, `zoom_factor=0.2` result in a random zoom
            factor from 80% to 120%. If left unspecified, it will be tuned
            automatically.
        contrast_factor: A positive float represented as fraction of value, or a
            tuple of size 2 representing lower and upper bound, or a
            kerastuner.engine.hyperparameters.Choice range of floats to find the
            optimal value. When represented as a single float, lower = upper.
            The contrast factor will be randomly picked
            between [1.0 - lower, 1.0 + upper]. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(
        self,
        translation_factor: Optional[
            Union[float, Tuple[float, float], hyperparameters.Choice]
        ] = None,
        vertical_flip: Optional[bool] = None,
        horizontal_flip: Optional[bool] = None,
        rotation_factor: Optional[Union[float, hyperparameters.Choice]] = None,
        zoom_factor: Optional[
            Union[float, Tuple[float, float], hyperparameters.Choice]
        ] = None,
        contrast_factor: Optional[
            Union[float, Tuple[float, float], hyperparameters.Choice]
        ] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.translation_factor = utils.get_hyperparameter(
            translation_factor,
            hyperparameters.Choice("translation_factor", [0.0, 0.1]),
            Union[float, Tuple[float, float]],
        )
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_factor = utils.get_hyperparameter(
            rotation_factor,
            hyperparameters.Choice("rotation_factor", [0.0, 0.1]),
            float,
        )
        self.zoom_factor = utils.get_hyperparameter(
            zoom_factor,
            hyperparameters.Choice("zoom_factor", [0.0, 0.1]),
            Union[float, Tuple[float, float]],
        )
        self.contrast_factor = utils.get_hyperparameter(
            contrast_factor,
            hyperparameters.Choice("contrast_factor", [0.0, 0.1]),
            Union[float, Tuple[float, float]],
        )

    @staticmethod
    def _get_fraction_value(value):
        if isinstance(value, tuple):
            return value
        return value, value

    def build(self, hp, inputs=None):
        input_node = tree.flatten(inputs)[0]
        output_node = input_node

        # Translate
        translation_factor = utils.add_to_hp(self.translation_factor, hp)
        if translation_factor not in [0, (0, 0)]:
            height_factor, width_factor = self._get_fraction_value(
                translation_factor
            )
            output_node = layers.RandomTranslation(height_factor, width_factor)(
                output_node
            )

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
            output_node = layers.RandomFlip(mode=flip_mode)(output_node)

        # Rotate
        rotation_factor = utils.add_to_hp(self.rotation_factor, hp)
        if rotation_factor != 0:
            output_node = layers.RandomRotation(rotation_factor)(output_node)

        # Zoom
        zoom_factor = utils.add_to_hp(self.zoom_factor, hp)
        if zoom_factor not in [0, (0, 0)]:
            height_factor, width_factor = self._get_fraction_value(zoom_factor)
            # TODO: Add back RandomZoom when it is ready.
            # output_node = layers.RandomZoom(
            # height_factor, width_factor)(output_node)

        # Contrast
        contrast_factor = utils.add_to_hp(self.contrast_factor, hp)
        if contrast_factor not in [0, (0, 0)]:
            output_node = layers.RandomContrast(contrast_factor)(output_node)

        return output_node

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "translation_factor": io_utils.serialize_block_arg(
                    self.translation_factor
                ),
                "horizontal_flip": self.horizontal_flip,
                "vertical_flip": self.vertical_flip,
                "rotation_factor": io_utils.serialize_block_arg(
                    self.rotation_factor
                ),
                "zoom_factor": io_utils.serialize_block_arg(self.zoom_factor),
                "contrast_factor": io_utils.serialize_block_arg(
                    self.contrast_factor
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["translation_factor"] = io_utils.deserialize_block_arg(
            config["translation_factor"]
        )
        config["rotation_factor"] = io_utils.deserialize_block_arg(
            config["rotation_factor"]
        )
        config["zoom_factor"] = io_utils.deserialize_block_arg(
            config["zoom_factor"]
        )
        config["contrast_factor"] = io_utils.deserialize_block_arg(
            config["contrast_factor"]
        )
        return cls(**config)
