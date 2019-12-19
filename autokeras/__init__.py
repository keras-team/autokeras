from packaging.version import parse
import tensorflow

if parse(tensorflow.__version__) < parse('2.0.0'):
    raise ImportError(
        f'The Tensorflow package version needs to be at least v2.0.0\n'
        f'for AutoKeras to run. Currently, your TensorFlow version is \n'
        f'v{tensorflow.__version__}. Please upgrade with '
        f'`pip install --upgrade tensorflow` for the GPU version, or use'
        f'`pip install --upgrade tensorflow-cpu` for the CPU version.'
        f'You can use `pip freeze` to check afterwards that everything is ok.'
    )

from autokeras.auto_model import AutoModel
from autokeras.const import Constant
from autokeras.hypermodel.base import Block
from autokeras.hypermodel.base import Head
from autokeras.hypermodel.base import HyperBlock
from autokeras.hypermodel.base import Node
from autokeras.hypermodel.base import Preprocessor
from autokeras.hypermodel.block import ConvBlock
from autokeras.hypermodel.block import DenseBlock
from autokeras.hypermodel.block import EmbeddingBlock
from autokeras.hypermodel.block import Merge
from autokeras.hypermodel.block import ResNetBlock
from autokeras.hypermodel.block import RNNBlock
from autokeras.hypermodel.block import SpatialReduction
from autokeras.hypermodel.block import TemporalReduction
from autokeras.hypermodel.block import XceptionBlock
from autokeras.hypermodel.head import ClassificationHead
from autokeras.hypermodel.head import RegressionHead
from autokeras.hypermodel.hyperblock import ImageBlock
from autokeras.hypermodel.hyperblock import StructuredDataBlock
from autokeras.hypermodel.hyperblock import TextBlock
from autokeras.hypermodel.node import ImageInput
from autokeras.hypermodel.node import Input
from autokeras.hypermodel.node import StructuredDataInput
from autokeras.hypermodel.node import TextInput
from autokeras.hypermodel.preprocessor import FeatureEngineering
from autokeras.hypermodel.preprocessor import ImageAugmentation
from autokeras.hypermodel.preprocessor import LightGBM
from autokeras.hypermodel.preprocessor import Normalization
from autokeras.hypermodel.preprocessor import TextToIntSequence
from autokeras.hypermodel.preprocessor import TextToNgramVector
from autokeras.task import ImageClassifier
from autokeras.task import ImageRegressor
from autokeras.task import StructuredDataClassifier
from autokeras.task import StructuredDataRegressor
from autokeras.task import TextClassifier
from autokeras.task import TextRegressor
