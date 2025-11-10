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

from autokeras import hyper_preprocessors
from autokeras import preprocessors


def test_serialize_and_deserialize_default_hpps():
    preprocessor = preprocessors.AddOneDimension()
    hyper_preprocessor = hyper_preprocessors.DefaultHyperPreprocessor(
        preprocessor
    )
    hyper_preprocessor = hyper_preprocessors.deserialize(
        hyper_preprocessors.serialize(hyper_preprocessor)
    )
    assert isinstance(
        hyper_preprocessor.preprocessor, preprocessors.AddOneDimension
    )


def test_serialize_and_deserialize_default_hpps_categorical():
    x_train = np.array([["a", "ab", 2.1], ["b", "bc", 1.0], ["a", "bc", "nan"]])
    preprocessor = preprocessors.CategoricalToNumerical(
        column_names=["column_a", "column_b", "column_c"],
        column_types={
            "column_a": "categorical",
            "column_b": "categorical",
            "column_c": "numerical",
        },
    )

    hyper_preprocessor = hyper_preprocessors.DefaultHyperPreprocessor(
        preprocessor
    )
    hyper_preprocessor.preprocessor.fit(x_train)
    hyper_preprocessor = hyper_preprocessors.deserialize(
        hyper_preprocessors.serialize(hyper_preprocessor)
    )
    assert isinstance(
        hyper_preprocessor.preprocessor,
        preprocessors.CategoricalToNumerical,
    )

    results = hyper_preprocessor.preprocessor.transform(x_train)

    assert results[0][0] == results[2][0]
    assert results[0][0] != results[1][0]
    assert results[0][1] != results[1][1]
    assert results[0][1] != results[2][1]
    assert results[2][2] == 0
    assert results.dtype == "float32"
