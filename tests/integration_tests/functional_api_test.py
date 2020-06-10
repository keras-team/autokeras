from tensorflow.python.keras.datasets import mnist

import autokeras as ak
from tests import utils


def test_functional_api(tmp_path):
    # Prepare the data.
    num_instances = 80
    (image_x, train_y), (test_x, test_y) = mnist.load_data()
    (text_x, train_y), (test_x, test_y) = utils.imdb_raw()
    (structured_data_x, train_y), (test_x, test_y) = utils.dataframe_numpy()

    image_x = image_x[:num_instances]
    text_x = text_x[:num_instances]
    structured_data_x = structured_data_x[:num_instances]
    classification_y = utils.generate_one_hot_labels(num_instances=num_instances,
                                                     num_classes=3)
    regression_y = utils.generate_data(num_instances=num_instances, shape=(1,))

    # Build model and train.
    image_input = ak.ImageInput()
    output = ak.Normalization()(image_input)
    output = ak.ImageAugmentation()(output)
    outputs1 = ak.ResNetBlock(version='next')(output)
    outputs2 = ak.XceptionBlock()(output)
    image_output = ak.Merge()((outputs1, outputs2))

    structured_data_input = ak.StructuredDataInput()
    structured_data_output = ak.CategoricalToNumerical()(structured_data_input)
    structured_data_output = ak.DenseBlock()(structured_data_output)

    text_input = ak.TextInput()
    outputs1 = ak.TextToIntSequence()(text_input)
    outputs1 = ak.Embedding()(outputs1)
    outputs1 = ak.ConvBlock(separable=True)(outputs1)
    outputs1 = ak.SpatialReduction()(outputs1)
    outputs2 = ak.TextToNgramVector()(text_input)
    outputs2 = ak.DenseBlock()(outputs2)
    text_output = ak.Merge()((
        outputs1,
        outputs2
    ))

    merged_outputs = ak.Merge()((
        structured_data_output,
        image_output,
        text_output
    ))

    regression_outputs = ak.RegressionHead()(merged_outputs)
    classification_outputs = ak.ClassificationHead()(merged_outputs)
    automodel = ak.AutoModel(
        inputs=[
            image_input,
            text_input,
            structured_data_input
        ],
        directory=tmp_path,
        outputs=[
            regression_outputs,
            classification_outputs
        ],
        max_trials=2,
        tuner=ak.Hyperband,
        seed=utils.SEED)

    automodel.fit(
        (
            image_x,
            text_x,
            structured_data_x
        ),
        (
            regression_y,
            classification_y
        ),
        validation_split=0.2,
        epochs=1)
