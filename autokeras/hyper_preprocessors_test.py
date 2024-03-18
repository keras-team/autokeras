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
