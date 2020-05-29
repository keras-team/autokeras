import tensorflow as tf

from autokeras.hypermodels.basic import ConvBlock
from autokeras.hypermodels.basic import DenseBlock
from autokeras.hypermodels.basic import Embedding
from autokeras.hypermodels.basic import ResNetBlock
from autokeras.hypermodels.basic import RNNBlock
from autokeras.hypermodels.basic import XceptionBlock
from autokeras.hypermodels.heads import ClassificationHead
from autokeras.hypermodels.heads import RegressionHead
from autokeras.hypermodels.heads import SegmentationHead
from autokeras.hypermodels.preprocessing import CategoricalToNumerical
from autokeras.hypermodels.preprocessing import ImageAugmentation
from autokeras.hypermodels.preprocessing import Normalization
from autokeras.hypermodels.preprocessing import TextToIntSequence
from autokeras.hypermodels.preprocessing import TextToNgramVector
from autokeras.hypermodels.reduction import Flatten
from autokeras.hypermodels.reduction import Merge
from autokeras.hypermodels.reduction import SpatialReduction
from autokeras.hypermodels.reduction import TemporalReduction
from autokeras.hypermodels.wrapper import ImageBlock
from autokeras.hypermodels.wrapper import StructuredDataBlock
from autokeras.hypermodels.wrapper import TextBlock
from autokeras.hypermodels.wrapper import TimeseriesBlock


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='hypermodels')
