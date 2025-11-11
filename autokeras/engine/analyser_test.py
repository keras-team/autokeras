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

from autokeras.engine.analyser import Analyser


def test_analyser_update_unicode_string_dtype():
    analyser = Analyser()
    data = np.array(["hello", "world"], dtype="U10")

    analyser.update(data)

    assert analyser.dtype == "string"
    assert analyser.shape == [2]
    assert analyser.batch_size == 2
    assert analyser.num_samples == 2


def test_analyser_update_byte_string_dtype():
    analyser = Analyser()
    data = np.array([b"hello", b"world"], dtype="S10")

    analyser.update(data)

    assert analyser.dtype == "string"
    assert analyser.shape == [2]
    assert analyser.batch_size == 2
    assert analyser.num_samples == 2


def test_analyser_update_numeric_dtype():
    analyser = Analyser()
    data = np.array([1, 2, 3], dtype=np.int32)

    analyser.update(data)

    assert analyser.dtype == "int32"
    assert analyser.shape == [3]
    assert analyser.batch_size == 3
    assert analyser.num_samples == 3


def test_analyser_update_float_dtype():
    analyser = Analyser()
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    analyser.update(data)

    assert analyser.dtype == "float64"
    assert analyser.shape == [3]
    assert analyser.batch_size == 3
    assert analyser.num_samples == 3


def test_analyser_finalize_not_implemented():
    analyser = Analyser()

    with pytest.raises(NotImplementedError):
        analyser.finalize()
