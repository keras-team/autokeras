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


class Pipeline:
    # It should include the ConcretePreprocessors, the keras.Model, and the
    # ConcretePostprocessors.

    # This class should also be exportable. It is the default export format.
    @classmethod
    def from_state(cls, state):
        raise NotImplementedError
