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

from autokeras.engine import serializable


class Encoder(serializable.Serializable):
    """Base class for encoders of the prediction targets.

    # Arguments
        num_classes: Int. The number of classes. Defaults to None.
    """

    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self._labels = None
        self._int_to_label = {}

    def fit_with_labels(self, data):
        """Fit the encoder with all the labels.

        # Arguments
            data: numpy.ndarray. The original labels.
        """
        raise NotImplementedError

    def encode(self, data):
        """Encode the original labels.

        # Arguments
            data: numpy.ndarray. The original labels.

        # Returns
            numpy.ndarray. The encoded labels.
        """
        raise NotImplementedError

    def decode(self, data):
        """Decode the encoded labels to original labels.

        # Arguments
            data: numpy.ndarray. The encoded labels.

        # Returns
            numpy.ndarray. The original labels.
        """
        raise NotImplementedError

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "labels": self._labels,
            "int_to_label": self._int_to_label,
        }

    @classmethod
    def from_config(cls, config):
        labels = config.pop("labels")
        int_to_label = config.pop("int_to_label")
        obj = super().from_config(config)
        obj._labels = labels
        obj._int_to_label = int_to_label
        return obj

    def _check_num_classes(self):
        if self.num_classes != len(self._labels):
            raise ValueError(
                "Expect {num_classes} classes in the training targets, "
                "but got {len_labels} classes.".format(
                    num_classes=self.num_classes, len_labels=len(self._labels)
                )
            )
