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

from autokeras.engine import block
from autokeras.prototype import graph_state


class BaseBlock(block.Block):
    # Renaming the autokeras.engine.block to BaseBlock.
    # Not open for extension.
    pass

    def _build_wrapper(self, hp, *args, **kwargs):
        with graph_state.get_state().build_scope(self):
            with hp.name_scope(self.name):
                return super()._build_wrapper(hp, *args, **kwargs)
