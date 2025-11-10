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

from autokeras.engine import adapter as adapter_module


class HeadAdapter(adapter_module.Adapter):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def check(self, dataset):
        if not isinstance(dataset, np.ndarray):
            raise TypeError(
                f"Expect the target data of {self.name} to be"
                f" np.ndarray, but got {type(dataset)}."
            )


class ClassificationAdapter(HeadAdapter):
    pass


class RegressionAdapter(HeadAdapter):
    pass
