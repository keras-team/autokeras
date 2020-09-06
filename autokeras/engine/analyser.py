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


class Analyser(object):
    """Analyze the dataset. set the result back to the io hypermodels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = None
        self.dtype = None

    def update(self, data):
        if self.dtype is None:
            self.dtype = data.dtype
        if self.shape is None:
            self.shape = data.shape.as_list()

    def finalize(self):
        raise NotImplementedError
