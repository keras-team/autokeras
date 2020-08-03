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

from autokeras import encoders


def test_one_hot_encoder_deserialize_transforms_to_np():
    encoder = encoders.OneHotEncoder()
    encoder.fit(np.array(["a", "b", "a"]))

    encoder = encoders.deserialize(encoders.serialize(encoder))
    one_hot = encoder.encode(np.array(["a"]))

    assert np.array_equal(one_hot, [[1, 0]]) or np.array_equal(one_hot, [[0, 1]])


def test_one_hot_encoder_decode_to_same_string():
    encoder = encoders.OneHotEncoder()
    encoder.fit(np.array(["a", "b", "a"]))

    assert encoder.decode(encoder.encode(np.array(["a"])))[0] == "a"


def test_wrong_num_classes_error():
    encoder = encoders.OneHotEncoder(num_classes=3)

    with pytest.raises(ValueError) as info:
        encoder.fit(np.array(["a", "b", "a"]))

    assert "Expect 3 classes in the training targets" in str(info.value)
