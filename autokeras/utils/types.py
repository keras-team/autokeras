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

from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

DatasetType = Union[np.ndarray, tf.data.Dataset]
LossType = Union[str, Callable, Loss]
AcceptableMetric = Union[str, Callable, Metric]
MetricsType = Union[
    List[AcceptableMetric], List[List[AcceptableMetric]], Dict[str, AcceptableMetric]
]
