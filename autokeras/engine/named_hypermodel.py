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

import keras_tuner

from autokeras.backend import keras
from autokeras.engine import serializable
from autokeras.utils import utils


class NamedHyperModel(keras_tuner.HyperModel, serializable.Serializable):
    """

    # Arguments
        name: String. The name of the HyperModel. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: str = None, **kwargs):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
            name = utils.to_snake_case(name)
        super().__init__(name=name, **kwargs)

    def get_config(self):
        """Get the configuration of the preprocessor.

        # Returns
            A dictionary of configurations of the preprocessor.
        """
        return {"name": self.name, "tunable": self.tunable}
