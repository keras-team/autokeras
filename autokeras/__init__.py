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

from autokeras.auto_model import AutoModel
from autokeras.blocks import BertBlock
from autokeras.blocks import CategoricalToNumerical
from autokeras.blocks import ClassificationHead
from autokeras.blocks import ConvBlock
from autokeras.blocks import DenseBlock
from autokeras.blocks import EfficientNetBlock
from autokeras.blocks import Embedding
from autokeras.blocks import Flatten
from autokeras.blocks import ImageAugmentation
from autokeras.blocks import ImageBlock
from autokeras.blocks import Merge
from autokeras.blocks import Normalization
from autokeras.blocks import RegressionHead
from autokeras.blocks import ResNetBlock
from autokeras.blocks import RNNBlock
from autokeras.blocks import SpatialReduction
from autokeras.blocks import StructuredDataBlock
from autokeras.blocks import TemporalReduction
from autokeras.blocks import TextBlock
from autokeras.blocks import TextToIntSequence
from autokeras.blocks import TextToNgramVector
from autokeras.blocks import Transformer
from autokeras.blocks import XceptionBlock
from autokeras.engine.block import Block
from autokeras.engine.head import Head
from autokeras.engine.node import Node
from autokeras.keras_layers import BertEncoder
from autokeras.keras_layers import BertTokenizer
from autokeras.keras_layers import CastToFloat32
from autokeras.keras_layers import ExpandLastDim
from autokeras.keras_layers import MultiCategoryEncoding
from autokeras.nodes import ImageInput
from autokeras.nodes import Input
from autokeras.nodes import StructuredDataInput
from autokeras.nodes import TextInput
from autokeras.nodes import TimeseriesInput
from autokeras.tasks import ImageClassifier
from autokeras.tasks import ImageRegressor
from autokeras.tasks import StructuredDataClassifier
from autokeras.tasks import StructuredDataRegressor
from autokeras.tasks import TextClassifier
from autokeras.tasks import TextRegressor
from autokeras.tasks import TimeseriesForecaster
from autokeras.tuners import BayesianOptimization
from autokeras.tuners import Greedy
from autokeras.tuners import Hyperband
from autokeras.tuners import RandomSearch
from autokeras.utils.io_utils import image_dataset_from_directory
from autokeras.utils.io_utils import text_dataset_from_directory
from autokeras.utils.utils import check_kt_version
from autokeras.utils.utils import check_tf_version

__version__ = "1.0.11"

check_tf_version()
check_kt_version()

CUSTOM_OBJECTS = {
    "BertEncoder": BertEncoder,
    "BertTokenizer": BertTokenizer,
    "CastToFloat32": CastToFloat32,
    "ExpandLastDim": ExpandLastDim,
    "MultiCategoryEncoding": MultiCategoryEncoding,
}
