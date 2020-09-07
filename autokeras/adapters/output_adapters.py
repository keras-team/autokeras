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
import pandas as pd
import tensorflow as tf

from autokeras.engine import adapter as adapter_module


class HeadAdapter(adapter_module.Adapter):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def check(self, dataset):
        supported_types = (tf.data.Dataset, np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(dataset, supported_types):
            raise TypeError(
                "Expect the target data of {name} to be tf.data.Dataset,"
                " np.ndarray, pd.DataFrame or pd.Series, but got {type}.".format(
                    name=self.name, type=type(dataset)
                )
            )

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)


class ClassificationAdapter(HeadAdapter):
    pass


class RegressionAdapter(HeadAdapter):
    pass


class SegmentationHeadAdapter(ClassificationAdapter):
    pass
