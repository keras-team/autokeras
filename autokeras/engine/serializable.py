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


class Serializable(object):
    """Serializable from and to JSON with same mechanism as Keras Layer."""

    def get_config(self):
        """Returns the current config of this object.

        # Returns
            Dictionary.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        """Build an instance from the config of this object.

        # Arguments
            config: Dict. The config of the object.
        """
        return cls(**config)
