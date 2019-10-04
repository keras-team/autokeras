import pytest
from tensorflow.python.keras.datasets import mnist

import autokeras as ak
from tests import common


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_functional_api')


def test_functional_api(tmp_dir):
    # Prepare the data.
    num_instances = 20
    (image_x, train_y), (test_x, test_y) = mnist.load_data()
    (text_x, train_y), (test_x, test_y) = common.imdb_raw()
    (structured_data_x, train_y), (test_x, test_y) = common.dataframe_numpy()

    image_x = image_x[:num_instances]
    text_x = text_x[:num_instances]
    structured_data_x = structured_data_x[:num_instances]
    classification_y = common.generate_one_hot_labels(num_instances=num_instances,
                                                      num_classes=3)
    regression_y = common.generate_data(num_instances=num_instances, shape=(1,))

    # Build model and train.
    # image_input = ak.ImageInput()
    # output = ak.Normalization()(image_input)
    # output = ak.ImageAugmentation()(output)
    # outputs1 = ak.ResNetBlock(version='next')(image_input)
    # outputs2 = ak.XceptionBlock()(image_input)
    # image_output = ak.Merge()((outputs1, outputs2))

    # structured_data_input = ak.StructuredDataInput(
    #     column_names=common.COLUMN_NAMES_FROM_CSV,
    #     column_types=common.COLUMN_TYPES_FROM_CSV)
    # structured_data_output = ak.FeatureEngineering()(structured_data_input)
    # structured_data_output = ak.DenseBlock()(structured_data_output)

    text_input = ak.TextInput()
    outputs1 = ak.TextToIntSequence()(text_input)
    outputs1 = ak.EmbeddingBlock()(outputs1)
    # outputs1 = ak.ConvBlock(separable=True)(outputs1)
    # outputs1 = ak.SpatialReduction()(outputs1)
    # outputs2 = ak.TextToNgramVector()(text_input)
    # outputs2 = ak.DenseBlock()(outputs2)
    text_output = ak.Merge()((
        outputs1,
        # outputs2
        ))

    merged_outputs = ak.Merge()((
        # structured_data_output,
        # image_output,
        text_output
        ))

    regression_outputs = ak.RegressionHead()(merged_outputs)
    classification_outputs = ak.ClassificationHead()(merged_outputs)
    automodel = ak.GraphAutoModel(
        inputs=[
            # image_input,
            text_input,
            # structured_data_input
                ],
        directory=tmp_dir,
        outputs=[regression_outputs,
                 classification_outputs],
        max_trials=2,
        seed=common.SEED)

    automodel.fit(
        (
            # image_x,
            text_x,
            # structured_data_x
        ),
        (regression_y, classification_y),
        validation_split=0.2,
        epochs=2)
