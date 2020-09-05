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

import kerastuner


class HyperParameter(kerastuner.engine.hyperparameters.HyperParameter):
    def __init__(self, **kwargs):
        super().__init__(name="unknown", **kwargs)

    def add_to_hp(self, hp, name):
        kwargs = self.get_config()
        kwargs["name"] = name
        class_name = self.__class__.__name__
        func = getattr(hp, class_name)
        func(**kwargs)
