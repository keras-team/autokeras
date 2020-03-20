from typedapi import ensure_api_is_typed

import autokeras

HELP_MESSAGE = (
    "You can also take a look at this issue:\n"
    "https://github.com/keras-team/autokeras/issues/918"
)


# TODO: add types and remove all elements from
# the exception list.
EXCEPTION_LIST = [
    autokeras.Head,
    autokeras.Node,
    autokeras.CategoricalToNumerical,
    autokeras.Flatten,
    autokeras.ImageAugmentation,
    autokeras.ImageBlock,
    autokeras.Merge,
    autokeras.StructuredDataBlock,
    autokeras.TemporalReduction,
    autokeras.TextBlock,
    autokeras.TextToIntSequence,
    autokeras.TextToNgramVector,
    autokeras.ImageInput,
    autokeras.Input,
    autokeras.StructuredDataInput,
    autokeras.TimeseriesInput,
    autokeras.TextInput,
    autokeras.ImageClassifier,
    autokeras.ImageRegressor,
    autokeras.StructuredDataClassifier,
    autokeras.TextRegressor,
]


def test_api_surface_is_typed():
    ensure_api_is_typed(
        [autokeras], EXCEPTION_LIST, init_only=True, additional_message=HELP_MESSAGE,
    )
