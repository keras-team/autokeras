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

from autokeras.engine import analyser


class InputAnalyser(analyser.Analyser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = None

    def update(self, data):
        self.shape = data.shape.as_list()


class ImageAnalyser(InputAnalyser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_channel_dim = False

    def finalize(self):
        if len(self.shape) not in [3, 4]:
            raise ValueError(
                "Expect the data to ImageInput to have shape (batch_size, "
                "height, width, channels) or (batch_size, height, width) "
                "dimensions, but got input shape {shape}".format(shape=self.shape)
            )
        self.has_channel_dim = len(self.shape) == 4
