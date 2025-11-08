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

from autokeras.analysers import input_analysers


def test_image_input_analyser_shape_is_list_of_int():
    analyser = input_analysers.ImageAnalyser()
    dataset = np.random.rand(100, 32, 32, 3)

    analyser.update(dataset)
    analyser.finalize()

    assert isinstance(analyser.shape, list)
    assert all(map(lambda x: isinstance(x, int), analyser.shape))


def test_image_input_with_three_dim():
    analyser = input_analysers.ImageAnalyser()
    dataset = np.random.rand(100, 32, 32)

    analyser.update(dataset)
    analyser.finalize()

    assert len(analyser.shape) == 3


def test_image_input_with_illegal_dim():
    analyser = input_analysers.ImageAnalyser()
    dataset = np.random.rand(100, 32)

    with pytest.raises(ValueError) as info:
        analyser.update(dataset)
        analyser.finalize()

    assert "Expect the data to ImageInput to have shape" in str(info.value)


def test_text_input_with_illegal_dim():
    analyser = input_analysers.TextAnalyser()
    dataset = np.random.rand(100, 32)

    with pytest.raises(ValueError) as info:
        analyser.update(dataset)
        analyser.finalize()

    assert "Expect the data to TextInput to have shape" in str(info.value)


def test_text_analyzer_with_one_dim_doesnt_crash():
    analyser = input_analysers.TextAnalyser()
    dataset = np.array(["a b c", "b b c"])

    analyser.update(dataset)
    analyser.finalize()


def test_text_illegal_type_error():
    analyser = input_analysers.TextAnalyser()
    dataset = np.random.rand(100, 1)

    with pytest.raises(TypeError) as info:
        analyser.update(dataset)
        analyser.finalize()

    assert "Expect the data to TextInput to be strings" in str(info.value)
