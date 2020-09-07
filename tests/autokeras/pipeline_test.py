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

from autokeras import pipeline as pipeline_module
from autokeras import preprocessors


def test_pipeline_postprocess_one_hot_to_labels():
    pipeline = pipeline_module.Pipeline(
        inputs=[[]], outputs=[[preprocessors.OneHotEncoder(["a", "b", "c"])]]
    )
    assert np.array_equal(pipeline.postprocess(np.eye(3)), [["a"], ["b"], ["c"]])


def test_pipeline_postprocess_multiple_one_hot_to_labels():
    pipeline = pipeline_module.Pipeline(
        inputs=[[]],
        outputs=[
            [preprocessors.OneHotEncoder(["a", "b", "c"])],
            [preprocessors.OneHotEncoder(["a", "b", "c"])],
        ],
    )
    result = pipeline.postprocess([np.eye(3), np.eye(3)])
    assert np.array_equal(result[0], [["a"], ["b"], ["c"]])
    assert np.array_equal(result[1], [["a"], ["b"], ["c"]])
