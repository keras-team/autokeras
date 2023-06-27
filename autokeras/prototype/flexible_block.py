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

from autokeras.prototype import base_block


class FlexibleBlock(base_block.BaseBlock):
    # Open for extension.
    def _build_wrapper(self, hp, inputs, *args, **kwargs):
        # Accept both KerasTensor & Dataset.
        # If the inputs are datasets, build the block into a model and pass
        # through it.
        pass
