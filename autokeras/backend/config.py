# Copyright 2019 The AutoKeras Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras


def _multi_backend():
    version_fn = getattr(keras, "version", None)
    return version_fn and version_fn().startswith("3.")


_MULTI_BACKEND = _multi_backend()


def multi_backend():
    """Check if multi-backend keras is enabled."""
    return _MULTI_BACKEND


def backend():
    """Check the backend framework."""
    return "tensorflow" if not multi_backend() else keras.config.backend()
