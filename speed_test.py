import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

import autokeras as ak


def test_functional_api():
    # Prepare the data.
    (image_x, train_y), (test_x, test_y) = mnist.load_data()
    num_instances = len(image_x)

    image_x = image_x[:num_instances]
    classification_y = train_y[:num_instances]

    # Build model and train.
    image_input = ak.ImageInput()
    output = image_input
    output = ak.Normalization()(output)
    # output = ak.ImageAugmentation()(output)
    output = ak.ConvBlock()(output)

    classification_outputs = ak.ClassificationHead()(output)
    automodel = ak.GraphAutoModel(
        inputs=image_input,
        directory='.',
        outputs=classification_outputs,
        max_trials=1,
        seed=5)

    automodel.fit(
        image_x,
        classification_y,
        validation_split=0.2,
        epochs=1)

if __name__ == "__main__":
    test_functional_api()
