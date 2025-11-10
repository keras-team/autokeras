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

import numpy as np

from autokeras.engine import adapter as adapter_module


class InputAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "Expect the data to Input to be numpy.ndarray, "
                "but got {type}.".format(type=type(x))
            )
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError(
                "Expect the data to Input to be numerical, but got "
                "{type}.".format(type=x.dtype)
            )


class ImageAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "Expect the data to ImageInput to be numpy.ndarray, "
                "but got {type}.".format(type=type(x))
            )
        if isinstance(x, np.ndarray) and not np.issubdtype(x.dtype, np.number):
            raise TypeError(
                "Expect the data to ImageInput to be numerical, but got "
                "{type}.".format(type=x.dtype)
            )


class TextAdapter(adapter_module.Adapter):
    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "Expect the data to TextInput to be numpy.ndarray, "
                "but got {type}.".format(type=type(x))
            )


class StructuredDataAdapter(adapter_module.Adapter):
    def check(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "Unsupported type {type} for "
                "{name}.".format(type=type(x), name=self.__class__.__name__)
            )
