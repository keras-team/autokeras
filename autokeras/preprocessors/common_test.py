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

from autokeras import test_utils
from autokeras.preprocessors import common


def test_cast_to_int32_return_int32():
    x = test_utils.generate_one_hot_labels(100, 10)
    x = x.astype("uint8")
    x = common.CastToInt32().transform(x)
    assert x.dtype == "int32"
