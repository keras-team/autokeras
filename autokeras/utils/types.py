from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

AcceptableLoss = Union[str, Callable, Loss]
AcceptableMetric = Union[str, Callable, Metric]
AcceptableMetrics = Union[List[AcceptableMetric],
                          List[List[AcceptableMetric]],
                          Dict[str, AcceptableMetric],
                          None]
