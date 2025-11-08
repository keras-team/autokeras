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


class Dataset:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def map(self, func):
        mapped_data = func(self._data)
        return Dataset(mapped_data)

    @staticmethod
    def zip(datasets):
        return Dataset(list(zip(datasets)))

    def batch(self, batch_size):
        batched_data = [
            self._data[i : i + batch_size]
            for i in range(0, len(self._data), batch_size)
        ]
        return Dataset(batched_data)

    @staticmethod
    def from_tensor_slices(data):
        def convert_to_numpy(obj):
            """Recursively convert nested structures to numpy arrays."""
            if isinstance(obj, np.ndarray):
                # Already a numpy array
                return obj
            elif isinstance(obj, (list, tuple)):
                # Check if nested structure or flat list
                has_nested = len(obj) > 0 and any(
                    isinstance(item, (list, tuple, np.ndarray)) for item in obj
                )
                if has_nested:
                    # Nested structure - recursively convert elements
                    converted = type(obj)(
                        convert_to_numpy(item) for item in obj
                    )
                    return converted
                else:
                    # Flat list/tuple - convert to single array
                    return np.array(obj)
            else:
                # Convert to numpy array (handles scalars, etc.)
                return np.array(obj)

        return Dataset(convert_to_numpy(data))

    @property
    def data(self):
        return self._data
