
import autokeras
from typedapi import ensure_api_is_typed

HELP_MESSAGE = (
    "You can also take a look at this issue:\n"
    "https://github.com/keras-team/autokeras/issues/918"
)


# TODO: add types and remove all elements from
# the exception list.
EXCEPTION_LIST = [
    autokeras.AutoModel,
    autokeras.Block,
    autokeras.Head,
    autokeras.Node,
    autokeras.CategoricalToNumerical,
    autokeras.ClassificationHead,
    autokeras.ConvBlock,
    autokeras.DenseBlock,
    autokeras.Embedding,
    autokeras.Flatten,
    autokeras.ImageAugmentation,
    autokeras.ImageBlock,
    autokeras.Merge,
    autokeras.Normalization,
    autokeras.RegressionHead,
    autokeras.ResNetBlock,
    autokeras.RNNBlock,
    autokeras.SpatialReduction,
    autokeras.StructuredDataBlock,
    autokeras.TemporalReduction,
    autokeras.TextBlock,
    autokeras.TextToIntSequence,
    autokeras.TextToNgramVector,
    autokeras.XceptionBlock,
    autokeras.ImageInput,
    autokeras.Input,
    autokeras.StructuredDataInput,
    autokeras.TextInput,
    autokeras.ImageClassifier,
    autokeras.ImageRegressor,
    autokeras.StructuredDataClassifier,
    autokeras.StructuredDataRegressor,
    autokeras.TextClassifier,
    autokeras.TextRegressor
]


def test_api_surface_is_typed():
    ensure_api_is_typed(
        [autokeras], EXCEPTION_LIST, init_only=True, additional_message=HELP_MESSAGE,
    )
