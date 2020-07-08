import tensorflow as tf

from autokeras.blocks.basic import ConvBlock
from autokeras.blocks.basic import DenseBlock
from autokeras.blocks.basic import Embedding
from autokeras.blocks.basic import ResNetBlock
from autokeras.blocks.basic import RNNBlock
from autokeras.blocks.basic import Transformer
from autokeras.blocks.basic import XceptionBlock
from autokeras.blocks.heads import ClassificationHead
from autokeras.blocks.heads import RegressionHead
from autokeras.blocks.heads import SegmentationHead
from autokeras.blocks.preprocessing import CategoricalToNumerical
from autokeras.blocks.preprocessing import ImageAugmentation
from autokeras.blocks.preprocessing import Normalization
from autokeras.blocks.preprocessing import TextToIntSequence
from autokeras.blocks.preprocessing import TextToNgramVector
from autokeras.blocks.reduction import Flatten
from autokeras.blocks.reduction import Merge
from autokeras.blocks.reduction import SpatialReduction
from autokeras.blocks.reduction import TemporalReduction
from autokeras.blocks.wrapper import ImageBlock
from autokeras.blocks.wrapper import StructuredDataBlock
from autokeras.blocks.wrapper import TextBlock
from autokeras.blocks.wrapper import TimeseriesBlock


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='hypermodels')
