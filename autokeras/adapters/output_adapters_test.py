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
import pytest

from autokeras.adapters import output_adapters


def test_unsupported_types_error():
    adapter = output_adapters.ClassificationAdapter(name="a")

    with pytest.raises(TypeError) as info:
        adapter.adapt(1)

    assert "Expect the target data of a to be" in str(info.value)


def test_reg_head_transform_1d_np():
    adapter = output_adapters.RegressionAdapter(name="a")

    y = adapter.adapt(np.random.rand(10))

    assert isinstance(y, np.ndarray)
